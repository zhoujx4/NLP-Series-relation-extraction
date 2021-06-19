"""
@Time : 2020/12/1110:44
@Auth : 周俊贤
@File ：run.py
@DESCRIPTION:

"""
import copy
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from config.mpn import spo_config
from dataset.dataset_mhs import mhs_DuIEDataset, read_examples
from models.model_mhs_biaffine import model_mhs_biaffine
from run_evaluation import convert_spo_contour, convert2ressult, run_evaluate, convert_spo_contour2
from utils.bert_optimizaation import BertAdam
from utils.finetuning_argparse import get_argparse
from utils.utils import seed_everything, ProgressBar, init_logger, logger, write_prediction_results

def train(args, train_iter, model):
    logger.info("***** Running train *****")
    batch_loss = []
    batch_subject_loss = []
    batch_po_loss = []
    pbar = ProgressBar(n_total=len(train_iter), desc='Training')
    print("****" * 20)
    model.train()
    for step, batch in enumerate(train_iter):
        batch = tuple(t.to(args.device) for t in batch)
        batch_token_ids, batch_subject_type_ids, batch_subject_labels, batch_object_labels = batch
        loss, loss_sub, loss_rel = model(batch_token_ids=batch_token_ids,
                                         batch_subject_type_ids=batch_subject_type_ids,
                                         batch_subject_labels=batch_subject_labels,
                                         batch_object_labels=batch_object_labels)
        #
        loss.backward()
        args.optimizer.step()
        args.optimizer.zero_grad()

        #
        batch_loss.append(loss.item())
        batch_subject_loss.append(loss_sub.item())
        batch_po_loss.append(loss_rel.item())
        pbar(step,
             {'loss': sum(batch_loss[-20:]) / 20,
              'subject_loss': sum(batch_subject_loss[-20:]) / 20,
              'po_loss': sum(batch_po_loss[-20:]) / 20,
              })


def evaluate(args, eval_iter, model, mode):
    logger.info("***** Running Evalation *****")
    eval_file = eval_iter.dataset.examples
    answer_dict = {i: [[], [], {}] for i in range(len(eval_file))}

    pbar = ProgressBar(n_total=len(eval_iter), desc="Evaluating")
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_iter):
            batch = tuple(t.to(args.device) for t in batch)
            q_ids, batch_token_ids = batch
            end_list, subjects_str, output_logits = model(q_ids=q_ids,
                                                          batch_token_ids=batch_token_ids,
                                                          is_eval=True)
            convert_spo_contour2(qids=q_ids,
                                 end_list=end_list,
                                 subjects_str=subjects_str,
                                 output_logits=output_logits,
                                 eval_file=eval_file,
                                 answer_dict=answer_dict)
            pbar(step)

    convert2ressult(args, eval_file=eval_file, answer_dict=answer_dict)

    res = run_evaluate(eval_file, answer_dict, "dev")

    formatted_outputs = []
    # for example in eval_file:
    for p_id, example in enumerate(eval_file):
        # p_id = example.p_id
        d_record = {}
        d_record["text"] = example.context
        # d_record["spo_list"] = answer_dict[p_id - 1][1]
        d_record["spo_list"] = answer_dict[p_id][1]
        formatted_outputs.append(d_record)

    if mode == "test":
        predict_file_path = "./output/mpn_duie.json"
    else:
        predict_file_path = "./output/mpn_{}_predictions.json".format(mode)
    write_prediction_results(formatted_outputs, predict_file_path)

    return res


def main():
    args = get_argparse().parse_args()
    args.time = time.strftime("%m-%d_%H:%M", time.localtime())
    args.cache_data = "./data/mhs"
    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
    init_logger(log_file="./log/mhs_{}.log".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    seed_everything(args.seed)

    # 设置保存目录
    if not os.path.exists(args.output_dir):
        print('mkdir {}'.format(args.output))
        os.mkdir(args.output_dir)
    if not os.path.exists(args.cache_data):
        print('mkdir {}'.format(args.cache_data))
        os.makedirs(args.cache_data)

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # config
    args.spo_conf = spo_config.BAIDU_RELATION
    args.id2rel = {item: key for key, item in args.spo_conf.items()}
    args.rel2id = args.spo_conf
    #
    args.s2id = {}
    args.s_type = spo_config.SPO_TAG["subject_type"] + spo_config.SPO_TAG["object_type"]
    args.s_type = [x.split("_")[0] for x in args.s_type]
    i = 1
    args.s_type = list(set(args.s_type))
    args.s_type.sort(key=lambda x: x)
    for st in args.s_type:
        args.s2id[st] = i
        i += 1
    args.R_num = len(args.rel2id)
    args.E_num = len(args.s2id)

    # tokenizer
    args.tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)

    # Dataset & Dataloader
    train_dataset = mhs_DuIEDataset(args,
                                    examples=read_examples(args, json_file="./data/duie_train.json"),
                                    data_type="train")
    eval_dataset = mhs_DuIEDataset(args,
                                   examples=read_examples(args, json_file="./data/duie_dev.json"),
                                   data_type="dev")

    train_iter = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=args.per_gpu_train_batch_size,
                            collate_fn=train_dataset._create_collate_fn(),
                            num_workers=8)
    # for batch in train_iter:
    #     batch_token_ids, batch_subject_type_ids, batch_subject_labels, batch_object_labels = batch

    eval_iter = DataLoader(eval_dataset,
                           shuffle=False,
                           batch_size=args.per_gpu_eval_batch_size,
                           collate_fn=eval_dataset._create_collate_fn(),
                           num_workers=4)

    # model
    model = model_mhs_biaffine.from_pretrained(args.model_name_or_path,
                                                E_num=args.E_num,
                                                E_em_size=250,
                                                R_num=args.R_num)
    model.to(args.device)

    # 优化器
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    args.optimizer = BertAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              warmup=args.warmup_proportion,
                              t_total=(int(
                                  len(train_dataset) / args.per_gpu_train_batch_size) + 1) * args.num_train_epochs)
    args.scaler = torch.cuda.amp.GradScaler()

    # 训练
    best_f1 = 0
    early_stop = 0
    for epoch, _ in enumerate(range(int(args.num_train_epochs))):
        model.train()
        train(args, train_iter, model)
        # 每轮epoch在验证集上计算分数
        res_dev = evaluate(args, eval_iter, model, mode="eval")
        logger.info(
            "The F1-score is {}".format(res_dev['f1'])
        )
        if res_dev['f1'] >= best_f1:
            early_stop = 0
            best_f1 = res_dev['f1']
            logger.info("the best eval f1 is {:.4f}, saving model !!".format(best_f1))
            best_model = copy.deepcopy(model.module if hasattr(model, "module") else model)
            torch.save(best_model.state_dict(),
                       os.path.join(args.output_dir,
                                    "mpn_{}_{}.pkl".format(os.path.split(args.model_name_or_path)[1], args.time)),
                       _use_new_zipfile_serialization=False)
        else:
            early_stop += 1
            if early_stop == args.early_stop:
                logger.info("Early stop in {} epoch!".format(epoch))
                break


if __name__ == "__main__":
    main()

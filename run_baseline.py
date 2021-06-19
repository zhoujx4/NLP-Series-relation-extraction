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
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertTokenizerFast

from dataset.dataset_baseline import DuIEDataset, read_examples
from models.model_baseline import model_baseline
from run_evaluation import run_evaluate
from utils.finetuning_argparse import get_argparse
from utils.utils import seed_everything, ProgressBar, init_logger, logger, decoding, write_prediction_results


class BCELossForDuIE(nn.Module):
    def __init__(self, ):
        super(BCELossForDuIE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels.float())
        mask = mask.float()
        loss = loss * mask.unsqueeze(-1)
        loss = torch.sum(loss.mean(axis=2), axis=1) / torch.sum(mask, axis=1)
        loss = loss.mean()
        return loss


def train(args, train_iter, model):
    logger.info("***** Running train *****")
    # 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    linear_param_optimizer = list(model.fc1.named_parameters())
    linear_param_optimizer.extend(model.fc2.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.linear_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.linear_learning_rate},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    # 损失函数
    criterion = BCELossForDuIE().to(args.device)
    batch_loss = 0
    pbar = ProgressBar(n_total=len(train_iter), desc='Training')
    print("****" * 20)
    for step, batch in enumerate(train_iter):
        batch = tuple(t.to(args.device) for t in batch)
        batch_token_ids, batch_labels = batch
        mask = batch_token_ids != 0
        logits = model(
            input_ids=batch_token_ids,
            attention_mask=mask
        )

        # 正常训练
        loss = criterion(logits, batch_labels, mask)
        loss.backward()
        #
        batch_loss += loss.item()
        pbar(step,
             {
                 'batch_loss': batch_loss / (step + 1),
             })
        optimizer.step()
        model.zero_grad()


def evaluate(args, eval_iter, model, mode):
    logger.info("***** Running Evalation *****")
    eval_file = eval_iter.dataset.examples
    answer_dict = {i: [[], [], {}] for i in range(len(eval_file))}

    with open("config/官方baseline/id2spo.json", 'r', encoding='utf8') as fp:
        id2spo = json.load(fp)

    probs_all = []
    seq_len_all = [len(x.input_ids) for x in eval_iter.dataset.examples]
    offset_mapping_all = [x.offset_mapping for x in eval_iter.dataset.examples]

    pbar = ProgressBar(n_total=len(eval_iter), desc="Evaluating")
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_iter):
            pbar(step)
            batch = tuple(t.to(args.device) for t in batch)
            batch_token_ids, batch_labels = batch
            mask = batch_token_ids != 0
            logits = model(
                input_ids=batch_token_ids,
                attention_mask=mask
            )

            ###########
            probs = torch.sigmoid(logits)  # (B, L, N)
            probs_all.extend(list(probs.cpu().numpy()))

    decoding(eval_iter.dataset.examples,
             id2spo,
             probs_all,
             seq_len_all,
             offset_mapping_all,
             answer_dict)

    res = run_evaluate(eval_file, answer_dict, "dev")

    formatted_outputs = []
    for p_id, example in enumerate(eval_file):
        d_record = {}
        d_record["text"] = example.context
        d_record["spo_list"] = answer_dict[p_id][1]
        formatted_outputs.append(d_record)

    if mode == "test":
        predict_file_path = "./output/baseline_duie.json"
    else:
        predict_file_path = "./output/baseline_{}_predictions.json".format(mode)
    write_prediction_results(formatted_outputs, predict_file_path)

    return res

def main():
    args = get_argparse().parse_args()
    args.time = time.strftime("%m-%d_%H:%M", time.localtime())
    args.cache_data = "./data/baseline"
    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
    init_logger(log_file="./log/baseline_{}.log".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    seed_everything(args.seed)

    # 设置保存目录
    if not os.path.exists(args.output_dir):
        print('mkdir {}'.format(args.output))
        os.mkdir(args.output_dir)
    if not os.path.exists(args.cache_data):
        print('mkdir {}'.format(args.cache_data))
        os.makedirs(args.cache_data)

    # Reads label_map.
    with open("config/官方baseline/predicate2id.json", 'r', encoding='utf8') as fp:
        args.label_map = json.load(fp)
    num_classes = (len(args.label_map.keys()) - 2) * 2 + 2

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # tokenizer
    args.tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
    args.cls_token_id = args.tokenizer.cls_token_id
    args.sep_token_id = args.tokenizer.sep_token_id
    args.pad_token_id = args.tokenizer.pad_token_id

    # Dataset & Dataloader
    train_dataset = DuIEDataset(args,
                                examples=read_examples(args,
                                                       json_file="./data/duie_train.json"))
    eval_dataset = DuIEDataset(args,
                               examples=read_examples(args,
                                                      json_file="./data/duie_dev.json"))

    train_iter = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=args.per_gpu_train_batch_size,
                            collate_fn=train_dataset._create_collate_fn(),
                            num_workers=20)
    eval_iter = DataLoader(eval_dataset,
                           shuffle=False,
                           batch_size=args.per_gpu_eval_batch_size,
                           collate_fn=eval_dataset._create_collate_fn(),
                           num_workers=20)
    logger.info("The nums of the train_dataset features is {}".format(len(train_dataset)))
    logger.info("The nums of the eval_dataset features is {}".format(len(eval_dataset)))

    # model
    model = model_baseline(args.model_name_or_path, num_classes=num_classes)
    model.to(args.device)

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
                       os.path.join(args.output_dir, "baseline_{}_{}.pkl".format(os.path.split(args.model_name_or_path)[1], args.time)),
                       _use_new_zipfile_serialization=False)
        else:
            early_stop += 1
            if early_stop == args.early_stop:
                logger.info("Early stop in {} epoch!".format(epoch))
                break

if __name__ == "__main__":
    main()

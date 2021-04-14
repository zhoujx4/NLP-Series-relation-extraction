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

from dataset.dataset import DuIEDataset
from dataset.dataset import collate_fn
from models.model import DuIE_model
from utils.adversarial import FGM
from utils.finetuning_argparse import get_argparse
from utils.utils import seed_everything, ProgressBar, init_logger, logger, decoding, write_prediction_results, \
    get_precision_recall_f1

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
    linear_param_optimizer = list(model.classifier.named_parameters())
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
    fgm = FGM(model, epsilon=1, emb_name='word_embeddings.weight')
    for step, batch in enumerate(train_iter):
        for key in batch.keys():
            batch[key] = batch[key].to(args.device)
        logits = model(
            input_ids=batch['all_input_ids'],
            attention_mask=batch['all_attention_mask'],
            token_type_ids=batch['all_token_type_ids']
        )
        mask = (batch['all_input_ids'] != args.cls_token_id) & \
               (batch['all_input_ids'] != args.sep_token_id) & \
               (batch['all_input_ids'] != args.pad_token_id)

        # 正常训练
        loss = criterion(logits, batch["all_labels"], mask)
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

    with open("data/id2spo.json", 'r', encoding='utf8') as fp:
        id2spo = json.load(fp)

    probs_all = []
    seq_len_all = [x["seq_len"] for x in eval_iter.dataset.tokenized_examples]
    offset_mapping_all = [x["offset_mapping"] for x in eval_iter.dataset.tokenized_examples]
    batch_loss = 0

    pbar = ProgressBar(n_total=len(eval_iter), desc="Evaluating")
    model.eval()
    criterion = BCELossForDuIE().to(args.device)
    with torch.no_grad():
        for step, batch in enumerate(eval_iter):
            pbar(step)
            for key in batch.keys():
                batch[key] = batch[key].to(args.device)
            logits = model(
                input_ids=batch['all_input_ids'],
                attention_mask=batch['all_attention_mask'],
                token_type_ids=batch['all_token_type_ids']
            )

            mask = (batch['all_input_ids'] != args.cls_token_id) & \
                   (batch['all_input_ids'] != args.sep_token_id) & \
                   (batch['all_input_ids'] != args.pad_token_id)
            ###########
            loss = criterion(logits, batch["all_labels"], mask)
            batch_loss += loss.item()
            probs = torch.sigmoid(logits)  # (B, L, N)
            probs_all.extend(list(probs.cpu().numpy()))
    logger.info("The eval loss is {}".format(batch_loss))

    formatted_outputs = decoding(eval_iter.dataset.examples,
                                 id2spo,
                                 probs_all,
                                 seq_len_all,
                                 offset_mapping_all)


    predict_file_path = "./output/{}_predictions.json".format(mode)
    write_prediction_results(formatted_outputs, predict_file_path)

    if mode == "eval":
        precision, recall, f1 = get_precision_recall_f1("./data/duie_dev.json",
                                                        predict_file_path)
        return precision, recall, f1
    elif mode != "test":
        raise Exception("wrong mode for eval func")

    return

def main():
    args = get_argparse().parse_args()
    print(json.dumps(vars(args), sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))
    init_logger(log_file="./log/{}.log".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    seed_everything(args.seed)

    # 设置保存目录
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Reads label_map.
    with open("./data/predicate2id.json", 'r', encoding='utf8') as fp:
        label_map = json.load(fp)
    num_classes = (len(label_map.keys()) - 2) * 2 + 2

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
    args.cls_token_id = tokenizer.cls_token_id
    args.sep_token_id = tokenizer.sep_token_id
    args.pad_token_id = tokenizer.pad_token_id

    # Dataset & Dataloader
    train_dataset = DuIEDataset(args,
                                json_path="./data/duie_train.json",
                                tokenizer=tokenizer)
    eval_dataset = DuIEDataset(args,
                               json_path="./data/duie_dev.json",
                               tokenizer=tokenizer)
    # eval_dataset, test_dataset = random_split(eval_dataset,
    #                                           [round(0.5 * len(eval_dataset)),
    #                                            len(eval_dataset) - round(0.5 * len(eval_dataset))],
    #                                           generator=torch.Generator().manual_seed(42))
    train_iter = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=args.per_gpu_train_batch_size,
                            collate_fn=collate_fn,
                            num_workers=10)
    eval_iter = DataLoader(eval_dataset,
                           shuffle=False,
                           batch_size=args.per_gpu_eval_batch_size,
                           collate_fn=collate_fn,
                           num_workers=10)
    # test_iter = DataLoader(test_dataset,
    #                        shuffle=False,
    #                        batch_size=args.per_gpu_eval_batch_size,
    #                        collate_fn=collate_fn,
    #                        num_workers=10)
    logger.info("The nums of the train_dataset features is {}".format(len(train_dataset)))
    logger.info("The nums of the eval_dataset features is {}".format(len(eval_dataset)))

    # model
    model = DuIE_model(args.model_name_or_path, num_classes=num_classes)
    model.to(args.device)

    # 训练
    best_f1 = 0
    early_stop = 0
    for epoch, _ in enumerate(range(int(args.num_train_epochs))):
        model.train()
        train(args, train_iter, model)
        # 每轮epoch在验证集上计算分数
        if epoch > 2:
            eval_precision, eval_recall, eval_f1 = evaluate(args, eval_iter, model, mode="eval")
            print("precision: {:.4f}\n recall: {:.4f}\n f1: {:.4f}".format
                  (100 * eval_precision, 100 * eval_recall, 100 * eval_f1))

            logger.info(
                "The F1-score is {}".format(eval_f1)
            )
            if eval_f1 >= best_f1:
                early_stop = 0
                best_f1 = eval_f1
                logger.info("the best eval f1 is {:.4f}, saving model !!".format(best_f1))
                best_model = copy.deepcopy(model.module if hasattr(model, "module") else model)
                torch.save(best_model.state_dict(), os.path.join(args.output_dir, "best_model.pkl"))
            else:
                early_stop += 1
                if early_stop == args.early_stop:
                    logger.info("Early stop in {} epoch!".format(epoch))
                    break

if __name__ == "__main__":
    main()

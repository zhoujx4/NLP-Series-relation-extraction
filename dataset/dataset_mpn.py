"""
@Time : 2021/4/98:43
@Auth : 周俊贤
@File ：dataset.py
@DESCRIPTION:

"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.data_utils import covert_to_tokens, search_spo_index, search, Example, sequence_padding
from dataset.data_utils import save, load
from utils.utils import logger


def read_examples(args, json_file, data_type):
    examples_file = os.path.join(args.cache_data, os.path.splitext(os.path.split(json_file)[1])[0] + ".pkl")

    if not os.path.exists(examples_file):
        complex_relation_label = [6, 8, 24, 30, 44]
        complex_relation_affi_label = [7, 9, 25, 26, 27, 31, 45]
        examples = []
        with open(json_file, 'r') as fr:
            p_id = 0
            for line in tqdm(fr.readlines()):
                p_id += 1
                src_data = json.loads(line)
                text_raw = src_data['text']
                text_raw = text_raw.replace('®', '')
                text_raw = text_raw.replace('◆', '')
                tokens, tok_to_orig_start_index, tok_to_orig_end_index = covert_to_tokens(text_raw,
                                                                                          args.tokenizer,
                                                                                          return_orig_index=True)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                sub_po_dict, sub_ent_list, spo_list = dict(), list(), list()
                spoes = {}
                for spo in src_data.get('spo_list', []):
                    # spo_dict = dict()
                    for spo_object in spo['object'].keys():
                        if spo['predicate'] in args.spo_conf:
                            # 简单情况
                            predicate_label = args.spo_conf[spo['predicate']]

                            subject_sub_tokens = covert_to_tokens(spo['subject'],
                                                                  args.tokenizer,
                                                                  return_orig_index=False)
                            object_sub_tokens = covert_to_tokens(spo['object']['@value'],
                                                                 args.tokenizer,
                                                                 return_orig_index=False)
                            sub_ent_list.append(spo['subject'])
                        else:
                            # 复杂情况
                            predicate_label = args.spo_conf[spo['predicate'] + '_' + spo_object]
                            # 解决 N-ary 问题
                            if predicate_label in complex_relation_affi_label:
                                subject_sub_tokens = covert_to_tokens(spo['object']['@value'],
                                                                      args.tokenizer,
                                                                      return_orig_index=False)
                                sub_ent_list.append(spo['object']['@value'])
                            else:
                                subject_sub_tokens = covert_to_tokens(spo['subject'],
                                                                      args.tokenizer,
                                                                      return_orig_index=False)
                                sub_ent_list.append(spo['subject'])
                            object_sub_tokens = covert_to_tokens(spo['object'][spo_object],
                                                                 args.tokenizer,
                                                                 return_orig_index=False)

                        subject_start, object_start = search_spo_index(tokens, subject_sub_tokens, object_sub_tokens)

                        ###########################################
                        if subject_start == -1:
                            subject_start = search(subject_sub_tokens, tokens)
                        if object_start == -1:
                            object_start = search(object_sub_tokens, tokens)
                        ###########################################

                        if subject_start != -1 and object_start != -1:
                            s = (subject_start, subject_start + len(subject_sub_tokens) - 1)
                            o = (object_start, object_start + len(object_sub_tokens) - 1, predicate_label)
                            if s not in spoes:
                                spoes[s] = []
                            spoes[s].append(o)
                if data_type == "train":
                    for s, o in spoes.items():
                        tmp_spoes = {}
                        tmp_spoes[s] = spoes[s]
                        examples.append(
                            Example(
                                p_id=p_id,
                                context=text_raw,
                                tok_to_orig_start_index=tok_to_orig_start_index,
                                tok_to_orig_end_index=tok_to_orig_end_index,
                                bert_tokens=tokens,
                                sub_entity_list=sub_ent_list,
                                gold_answer=src_data.get('spo_list', []),
                                spoes=spoes,
                                tmp_spoes=tmp_spoes
                            ))

                else:
                    examples.append(
                        Example(
                            p_id=p_id,  # 1
                            context=text_raw,  # '《邪少兵王》是冰火未央写的网络小说连载于旗峰天下'
                            tok_to_orig_start_index=tok_to_orig_start_index,
                            # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
                            tok_to_orig_end_index=tok_to_orig_end_index,
                            # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
                            bert_tokens=tokens,
                            # ['[CLS]', '《', '邪', '少', '兵', '王', '》', '是', '冰', '火', '未', '央', '写', '的', '网', '络', '小', '说', '连', '载', '于', '旗', '峰', '天', '下', '[SEP]']
                            sub_entity_list=sub_ent_list,  # ['邪少兵王']
                            gold_answer=src_data.get('spo_list', []),
                            # [{'predicate': '作者', 'object_type': {'@value': '人物'}, 'subject_type': '图书作品', 'object': {'@value': '冰火未央'}, 'subject': '邪少兵王'}]
                            spoes=spoes  # {(2, 5): [(8, 11, 1)]}
                        ))
        save(examples_file, examples)
    else:
        logger.info('loading cache_data {}'.format(examples_file))
        examples = load(examples_file)
        logger.info('examples size is {}'.format(len(examples)))

    return examples


class mpn_DuIEDataset(Dataset):
    def __init__(self, args, examples, data_type):
        self.spo_config = args.spo_conf
        self.tokenizer = args.tokenizer
        self.max_len = args.max_len
        self.q_ids = list(range(len(examples)))
        self.examples = examples
        self.is_train = True if data_type == 'train' else False

    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, index):
        return self.q_ids[index], self.examples[index]

    def _create_collate_fn(self):
        def collate(examples):
            p_ids, examples = zip(*examples)
            p_ids = torch.tensor([p_id for p_id in p_ids], dtype=torch.long)
            batch_token_ids = []
            batch_subject_labels = []
            batch_subject_ids = []
            batch_object_labels = []
            for example in examples:
                spoes = example.spoes
                token_ids = self.tokenizer.encode(example.bert_tokens[1:-1],
                                                  is_split_into_words=False,
                                                  max_length=self.max_len,
                                                  truncation=True)  # TODO
                if self.is_train:
                    tmp_spoes = example.tmp_spoes
                    if spoes:
                        # subject标签
                        subject_labels = np.zeros((len(token_ids), 2), dtype=np.float32)
                        for s in spoes:
                            if s[1] <= self.max_len - 1:  # TODO
                                subject_labels[s[0], 0] = 1
                                subject_labels[s[1], 1] = 1
                        # ⚠️不是随机选一个subject
                        subject_ids = random.choice(list(tmp_spoes.keys()))
                        if subject_ids[1] >= self.max_len:
                            continue
                        # 对应的object标签
                        object_labels = np.zeros((len(token_ids), len(self.spo_config), 2), dtype=np.float32)
                        for o in spoes.get(subject_ids, []):
                            if o[1] <= self.max_len - 1:  # TODO
                                object_labels[o[0], o[2], 0] = 1
                                object_labels[o[1], o[2], 1] = 1

                        batch_token_ids.append(token_ids)
                        batch_subject_labels.append(subject_labels)
                        batch_subject_ids.append(subject_ids)
                        batch_object_labels.append(object_labels)
                else:
                    batch_token_ids.append(token_ids)
            if not batch_token_ids:
                a = 1
            batch_token_ids = sequence_padding(batch_token_ids, is_float=False)
            if not self.is_train:
                return p_ids, batch_token_ids
            else:
                batch_subject_ids = torch.tensor(batch_subject_ids)
                batch_subject_labels = sequence_padding(batch_subject_labels, padding=np.zeros(2), is_float=True)
                batch_object_labels = sequence_padding(batch_object_labels, padding=np.zeros((len(self.spo_config), 2)),
                                                       is_float=True)
                return batch_token_ids, batch_subject_ids, batch_subject_labels, batch_object_labels

        return collate


if __name__ == '__main__':
    pass

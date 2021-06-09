"""
@Time : 2021/4/98:43
@Auth : 周俊贤
@File ：dataset.py
@DESCRIPTION:

"""

import json
import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.data_utils import covert_to_tokens, search_spo_index, search, Example, sequence_padding, save, load
from utils.utils import logger

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
            batch_token_ids, batch_segment_ids = [], []
            batch_token_type_ids, batch_subject_type_ids, batch_subject_labels, batch_object_labels = [], [], [], []
            for example in examples:
                spoes = example.spoes
                token_ids = self.tokenizer.encode(example.bert_tokens[1:-1],
                                                  is_split_into_words=True,
                                                  max_length=self.max_len,
                                                  truncation=True)  # TODO
                segment_ids = len(token_ids) * [0]

                if self.is_train:
                    # subject标签
                    token_type_ids = np.zeros(len(token_ids), dtype=np.long)
                    subject_labels = np.zeros((len(token_ids), 2), dtype=np.float32)
                    for s in spoes:
                        if s[1] <= self.max_len:  # TODO
                            subject_labels[s[0], 0] = 1
                            subject_labels[s[1], 1] = 1
                    # ⚠️不是随机选一个subject
                    subject_ids = random.choice(list(spoes.keys()))
                    if subject_ids[1] > self.max_len:
                        break
                    # 对应的object标签
                    object_labels = np.zeros((len(token_ids), len(self.spo_config), 2), dtype=np.float32)
                    for o in spoes.get(subject_ids, []):
                        if o[1] <= self.max_len:  # TODO
                            object_labels[o[0], o[2], 0] = 1
                            object_labels[o[1], o[2], 1] = 1
                    batch_token_ids.append(token_ids)
                    batch_token_type_ids.append(token_type_ids)

                    batch_segment_ids.append(segment_ids)
                    batch_subject_labels.append(subject_labels)
                    batch_subject_ids.append(subject_ids)
                    batch_object_labels.append(object_labels)
                else:
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)

            batch_token_ids = sequence_padding(batch_token_ids, is_float=False)
            batch_segment_ids = sequence_padding(batch_segment_ids, is_float=False)
            if not self.is_train:
                return p_ids, batch_token_ids, batch_segment_ids
            else:
                batch_token_type_ids = sequence_padding(batch_token_type_ids, is_float=False)
                batch_subject_ids = torch.tensor(batch_subject_ids)
                batch_subject_labels = sequence_padding(batch_subject_labels, padding=np.zeros(2), is_float=True)
                batch_object_labels = sequence_padding(batch_object_labels, padding=np.zeros((len(self.spo_config), 2)),
                                                       is_float=True)
                ''' '''
                return batch_token_ids, batch_segment_ids, batch_token_type_ids, batch_subject_type_ids, batch_subject_labels, batch_object_labels

        return partial(collate)

if __name__ == '__main__':
    pass
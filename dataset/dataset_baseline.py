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
from transformers import BertTokenizerFast

from utils.finetuning_argparse import get_argparse
from utils.utils import logger, sequence_padding

class DuIEDataset(Dataset):
    def __init__(self, args, json_path, tokenizer):
        examples = []

        with open("./data/predicate2id.json", 'r', encoding='utf8') as fp:
            label_map = json.load(fp)

        with open(json_path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()

        examples = []
        tokenized_examples = []
        num_labels = 2 * (len(label_map.keys()) - 2) + 2

        cached_features_file = os.path.join(os.path.split(json_path)[0],
                                            '{}_cache'.format(os.path.split(json_path)[1]))
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            cached_data = torch.load(cached_features_file)
            examples = cached_data["examples"]
            tokenized_examples = cached_data["tokenized_examples"]
        else:
            for line in tqdm(lines):
                example = json.loads(line)
                # spo_list = example['spo_list'] if "spo_list" in example.keys() else None
                spo_list = example['spo_list'] if "spo_list" in example.keys() else []

                text_raw = example['text']

                #
                tokenized_example = tokenizer.encode_plus(
                    text_raw,
                    max_length=args.max_len,
                    padding="max_length",
                    truncation=True,
                    return_offsets_mapping=True,
                    return_length=True
                )
                #
                seq_len = sum(tokenized_example["attention_mask"])
                # seq_len = sum(tokenized_example["length"])
                tokens = tokenized_example["input_ids"]
                labels = [[0] * num_labels for i in range(args.max_len)]
                # labels = [[0] * num_labels for i in range(seq_len)]
                for spo in spo_list:
                    for spo_object in spo['object'].keys():
                        # assign relation label
                        if spo['predicate'] in label_map.keys():
                            # simple relation
                            label_subject = label_map[spo['predicate']]
                            label_object = label_subject + 55
                            subject_tokens = tokenizer.encode_plus(spo['subject'],
                                                                   add_special_tokens=False)["input_ids"]
                            object_tokens = tokenizer.encode_plus(spo['object']['@value'],
                                                                  add_special_tokens=False)[
                                "input_ids"]
                        else:
                            # complex relation
                            label_subject = label_map[spo['predicate'] + '_' + spo_object]
                            label_object = label_subject + 55
                            subject_tokens = tokenizer.encode_plus(spo['subject'],
                                                                   add_special_tokens=False)["input_ids"]
                            object_tokens = tokenizer.encode_plus(spo['object'][spo_object],
                                                                  add_special_tokens=False)[
                                "input_ids"]
                        subject_tokens_len = len(subject_tokens)
                        object_tokens_len = len(object_tokens)

                        # assign token label
                        # there are situations where s entity and o entity might overlap, e.g. xyz established xyz corporation
                        # to prevent single token from being labeled into two different entity
                        # we tag the longer entity first, then match the shorter entity within the rest text
                        forbidden_index = None
                        if subject_tokens_len > object_tokens_len:
                            for index in range(seq_len - subject_tokens_len + 1):
                                if tokens[index: index + subject_tokens_len] == subject_tokens:
                                    labels[index][label_subject] = 1
                                    for i in range(subject_tokens_len - 1):
                                        labels[index + i + 1][1] = 1
                                    forbidden_index = index
                                    break

                            for index in range(seq_len - object_tokens_len + 1):
                                if tokens[index: index + object_tokens_len] == object_tokens:
                                    if forbidden_index is None:
                                        labels[index][label_object] = 1
                                        for i in range(object_tokens_len - 1):
                                            labels[index + i + 1][1] = 1
                                        break
                                    # check if labeled already
                                    elif index < forbidden_index or index >= forbidden_index + len(subject_tokens):
                                        labels[index][label_object] = 1
                                        for i in range(object_tokens_len - 1):
                                            labels[index + i + 1][1] = 1
                                        break
                        else:
                            for index in range(seq_len - object_tokens_len + 1):
                                if tokens[index:index + object_tokens_len] == object_tokens:
                                    labels[index][label_object] = 1
                                    for i in range(object_tokens_len - 1):
                                        labels[index + i + 1][1] = 1
                                    forbidden_index = index
                                    break

                            for index in range(seq_len - subject_tokens_len + 1):
                                if tokens[index:index +
                                                subject_tokens_len] == subject_tokens:
                                    if forbidden_index is None:
                                        labels[index][label_subject] = 1
                                        for i in range(subject_tokens_len - 1):
                                            labels[index + i + 1][1] = 1
                                        break
                                    elif index < forbidden_index or index >= forbidden_index + len(
                                            object_tokens):
                                        labels[index][label_subject] = 1
                                        for i in range(subject_tokens_len - 1):
                                            labels[index + i + 1][1] = 1
                                        break
                for i in range(seq_len):
                    if labels[i] == [0] * num_labels:
                        labels[i][0] = 1
                tokenized_example["labels"] = labels
                tokenized_example["seq_len"] = seq_len

                examples.append(example)
                tokenized_examples.append(tokenized_example)
            # logger.info("Saving features into cached file %s", cached_features_file)
            # torch.save({"examples":examples, "tokenized_examples":tokenized_examples}, cached_features_file)

        self.examples = examples
        self.tokenized_examples = tokenized_examples

    def __len__(self):
        return len(self.tokenized_examples)

    def __getitem__(self, index):
        return self.tokenized_examples[index]

def collate_fn(batch):
    # max_len = max([sum(x['attention_mask']) for x in batch])
    # all_input_ids = pad_sequence([torch.tensor(x["input_ids"]) for x in batch], batch_first=True)
    # all_token_type_ids = pad_sequence([torch.tensor(x["token_type_ids"]) for x in batch], batch_first=True)
    # all_attention_mask = pad_sequence([torch.tensor(x["attention_mask"]) for x in batch], batch_first=True)
    # all_labels = pad_sequence([torch.tensor(x["labels"]) for x in batch], batch_first=True)

    max_len = max([sum(x['attention_mask']) for x in batch])
    all_input_ids = torch.tensor([x['input_ids'][:max_len] for x in batch])
    all_token_type_ids = torch.tensor([x['token_type_ids'][:max_len] for x in batch])
    all_attention_mask = torch.tensor([x['attention_mask'][:max_len] for x in batch])
    all_labels = torch.tensor([x["labels"][:max_len] for x in batch])

    return {
        "all_input_ids": all_input_ids,
        "all_token_type_ids": all_token_type_ids,
        "all_attention_mask": all_attention_mask,
        "all_labels": all_labels,
    }

if __name__ == '__main__':
    args = get_argparse().parse_args()
    tokenizer = BertTokenizerFast.from_pretrained("/data/zhoujx/prev_trained_model/rbt3")
    dataset = DuIEDataset(args, "../data/duie_train.json", tokenizer)
    a = 1
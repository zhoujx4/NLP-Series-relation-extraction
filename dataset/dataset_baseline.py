"""
@Time : 2021/4/98:43
@Auth : 周俊贤
@File ：dataset.py
@DESCRIPTION:

"""

import json
import os
from functools import partial

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast

from dataset.data_utils import load, save, sequence_padding
from utils.finetuning_argparse import get_argparse
from utils.utils import logger


class Example(object):
    def __init__(self,
                 p_id=None,
                 context=None,
                 offset_mapping=None,
                 input_ids=None,
                 entity_list=None,
                 gold_answer=None,
                 labels=None):
        self.p_id = p_id
        self.context = context
        self.offset_mapping = offset_mapping
        self.input_ids = input_ids
        self.entity_list = entity_list
        self.gold_answer = gold_answer
        self.labels = labels


def read_examples(args, json_file):
    examples_file = os.path.join(args.cache_data, os.path.splitext(os.path.split(json_file)[1])[0] + ".pkl")

    if not os.path.exists(examples_file):
        with open(json_file, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
        examples = []
        num_labels = 2 * (len(args.label_map.keys()) - 2) + 2

        for line in tqdm(lines):
            example = json.loads(line)
            ent_list = []
            text_raw = example['text']
            spo_list = example['spo_list'] if "spo_list" in example.keys() else []
            #
            tokenized_example = args.tokenizer.encode_plus(
                text_raw,
                # max_length=args.max_len,
                # padding="max_length",
                # truncation=True,
                return_offsets_mapping=True,
                # return_length=True
            )
            #
            seq_len = sum(tokenized_example["attention_mask"])
            tokens = tokenized_example["input_ids"]
            labels = [[0] * num_labels for i in range(len(tokens))]
            for spo in spo_list:
                for spo_object in spo['object'].keys():
                    if spo['predicate'] in args.label_map.keys():
                        # 简单知识
                        label_subject = args.label_map[spo['predicate']]
                        label_object = label_subject + 55
                        subject_tokens = args.tokenizer.encode_plus(spo['subject'], add_special_tokens=False)[
                            "input_ids"]
                        object_tokens = args.tokenizer.encode_plus(spo['object']['@value'], add_special_tokens=False)[
                            "input_ids"]
                        ent_list.append(spo['object']['@value'])
                    else:
                        # 复杂知识
                        label_subject = args.label_map[spo['predicate'] + '_' + spo_object]
                        label_object = label_subject + 55
                        subject_tokens = args.tokenizer.encode_plus(spo['subject'], add_special_tokens=False)[
                            "input_ids"]
                        object_tokens = args.tokenizer.encode_plus(spo['object'][spo_object], add_special_tokens=False)[
                            "input_ids"]
                        ent_list.append(spo['object'][spo_object])
                    ent_list.append(spo['subject'])
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
            for i in range(len(tokens)):
                if labels[i] == [0] * num_labels:
                    labels[i][0] = 1

            examples.append(
                Example(
                    context=text_raw,
                    offset_mapping=tokenized_example["offset_mapping"],
                    input_ids=tokenized_example["input_ids"],
                    entity_list=ent_list,
                    gold_answer=spo_list,
                    labels=labels
                ))
        save(examples_file, examples)
    else:
        logger.info('loading cache_data {}'.format(examples_file))
        examples = load(examples_file)
        logger.info('examples size is {}'.format(len(examples)))

    return examples


class DuIEDataset(Dataset):
    def __init__(self, args, examples):
        self.examples = examples
        self.max_len = args.max_len
        self.num_labels = 2 * (len(args.label_map.keys()) - 2) + 2

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].input_ids, self.examples[index].labels

    def _create_collate_fn(self):
        def collate(batch):
            input_ids, labels = zip(*batch)

            max_len = max([len(x) for x in input_ids])
            if max_len > self.max_len:
                max_len = self.max_len
            batch_token_ids = sequence_padding(input_ids)
            batch_token_ids = batch_token_ids[:, :max_len]
            batch_labels = [label[:max_len] if len(label) > max_len
                            else label + [[0] * self.num_labels] * (max_len - len(label))
                            for label in labels]
            batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            return batch_token_ids, batch_labels

        return partial(collate)


if __name__ == '__main__':
    args = get_argparse().parse_args()
    tokenizer = BertTokenizerFast.from_pretrained("/data/zhoujx/prev_trained_model/rbt3")
    dataset = DuIEDataset(args, "../data/duie_train.json", tokenizer)

"""
@Time : 2021/6/910:53
@Auth : 周俊贤
@File ：data_utils.py
@DESCRIPTION:

"""
import logging
import os
import pickle
import re
import sys

import torch
import numpy as np

from utils.utils import chineseandpunctuationextractor

class Example(object):
    def __init__(self,
                 p_id=None,
                 context=None,
                 tok_to_orig_start_index=None,
                 tok_to_orig_end_index=None,
                 bert_tokens=None,
                 spoes=None,
                 tmp_spoes=None,
                 sub_entity_list=None,
                 gold_answer=None, ):
        self.p_id = p_id
        self.context = context
        self.tok_to_orig_start_index = tok_to_orig_start_index
        self.tok_to_orig_end_index = tok_to_orig_end_index
        self.bert_tokens = bert_tokens
        self.spoes = spoes
        self.tmp_spoes = tmp_spoes
        self.sub_entity_list = sub_entity_list
        self.gold_answer = gold_answer

def save(filepath, obj, message=None):
    if message is not None:
        logging.info("Saving {}...".format(message))
    pickle_dump_large_file(obj, filepath)

def load(filepath):
    return pickle_load_large_file(filepath)

def pickle_dump_large_file(obj, filepath):
    max_bytes = 2 ** 31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])

def pickle_load_large_file(filepath):
    max_bytes = 2 ** 31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj

# def covert_to_tokens(text, tokenizer, max_seq_length, return_orig_index=False):
def covert_to_tokens(text, tokenizer, return_orig_index=False):
    sub_text = []
    buff = ""
    flag_en = False
    flag_digit = False
    for char in text:
        if chineseandpunctuationextractor.is_chinese_or_punct(char):
            if buff != "":
                sub_text.append(buff)
                buff = ""
            sub_text.append(char)
            flag_en = False
            flag_digit = False
        else:
            if re.compile('\d').match(char):
                if buff != "" and flag_en:
                    sub_text.append(buff)
                    buff = ""
                    flag_en = False
                flag_digit = True
                buff += char
            else:
                if buff != "" and flag_digit:
                    sub_text.append(buff)
                    buff = ""
                    flag_digit = False
                flag_en = True
                buff += char
    if buff != "":
        sub_text.append(buff)

    tok_to_orig_start_index = []
    tok_to_orig_end_index = []
    tokens = []
    text_tmp = ''
    for (i, token) in enumerate(sub_text):
        sub_tokens = tokenizer.tokenize(token) if token != ' ' else []
        text_tmp += token
        for sub_token in sub_tokens:
            tok_to_orig_start_index.append(len(text_tmp) - len(token))
            tok_to_orig_end_index.append(len(text_tmp) - 1)
            tokens.append(sub_token)
    if return_orig_index:
        return tokens, tok_to_orig_start_index, tok_to_orig_end_index
    else:
        return tokens

def search_spo_index(tokens, subject_sub_tokens, object_sub_tokens):
    subject_start_index, object_start_index = -1, -1
    forbidden_index = None
    if len(subject_sub_tokens) > len(object_sub_tokens):
        for index in range(
                len(tokens) - len(subject_sub_tokens) + 1):
            if tokens[index:index + len(
                    subject_sub_tokens)] == subject_sub_tokens:
                subject_start_index = index
                forbidden_index = index
                break

        for index in range(
                len(tokens) - len(object_sub_tokens) + 1):
            if tokens[index:index + len(
                    object_sub_tokens)] == object_sub_tokens:
                if forbidden_index is None:
                    object_start_index = index
                    break
                # check if labeled already
                elif index < forbidden_index or index >= forbidden_index + len(
                        subject_sub_tokens):
                    object_start_index = index

                    break

    else:
        for index in range(
                len(tokens) - len(object_sub_tokens) + 1):
            if tokens[index:index + len(
                    object_sub_tokens)] == object_sub_tokens:
                object_start_index = index
                forbidden_index = index
                break

        for index in range(
                len(tokens) - len(subject_sub_tokens) + 1):
            if tokens[index:index + len(
                    subject_sub_tokens)] == subject_sub_tokens:
                if forbidden_index is None:
                    subject_start_index = index
                    break
                elif index < forbidden_index or index >= forbidden_index + len(
                        object_sub_tokens):
                    subject_start_index = index
                    break

    return subject_start_index, object_start_index

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def sequence_padding(inputs, length=None, padding=0, is_float=False):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    outputs = np.array([
        np.concatenate([x, [padding] * (length - len(x))])
        if len(x) < length else x[:length] for x in inputs
    ])

    out_tensor = torch.FloatTensor(outputs) if is_float \
        else torch.LongTensor(outputs)
    return out_tensor
    # return torch.tensor(out_tensor)

def batch_gather(data: torch.Tensor, index: torch.Tensor):
    length = index.shape[0]
    t_index = index.cpu().numpy()
    t_data = data.cpu().data.numpy()
    result = []
    for i in range(length):
        result.append(t_data[i, t_index[i], :])  # 这里应该可以用一个更优雅的 api 来代替？

    return torch.from_numpy(np.array(result)).to(data.device)
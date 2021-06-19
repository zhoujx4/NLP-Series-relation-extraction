"""
@Time : 2021/2/78:58
@Auth : 周俊贤
@File ：utils.py
@DESCRIPTION:

"""

import codecs
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from utils import extract_chinese_and_punct

chineseandpunctuationextractor = extract_chinese_and_punct.ChineseAndPunctuationExtractor()

class Example(object):
    def __init__(self,
                 p_id=None,
                 context=None,
                 tok_to_orig_start_index=None,
                 tok_to_orig_end_index=None,
                 bert_tokens=None,
                 spoes=None,
                 sub_entity_list=None,
                 gold_answer=None, ):
        self.p_id = p_id
        self.context = context
        self.tok_to_orig_start_index = tok_to_orig_start_index
        self.tok_to_orig_end_index = tok_to_orig_end_index
        self.bert_tokens = bert_tokens
        self.spoes = spoes
        self.sub_entity_list = sub_entity_list
        self.gold_answer = gold_answer

def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''

    def __init__(self, n_total, width=30, desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')


logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    '''
    Example:
        >>> init_logger(log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)
    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        # file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def find_entity(text_raw,
                id_,
                predictions,
                offset_mapping):
    """
    retrieval entity mention under given predicate id for certain prediction.
    this is called by the "decoding" func.
    """
    entity_list = []
    for i in range(len(predictions)):
        if [id_] in predictions[i]:
            j = 0
            while i + j + 1 < len(predictions):  # 超过序列的长度
                if [1] in predictions[i + j + 1]:
                    j += 1
                else:
                    break
            entity = ''.join(text_raw[offset_mapping[i][0]:
                                      offset_mapping[i + j][1]])
            entity_list.append(entity)
    return list(set(entity_list))

def decoding(example_all,
             id2spo,
             logits_all,
             seq_len_all,
             offset_mapping_all,
             answer_dict
             ):
    """
    model output logits -> formatted spo (as in data set file)
    """
    for (i, (example, logits, seq_len, offset_mapping)) in \
            enumerate(zip(example_all, logits_all, seq_len_all, offset_mapping_all)):
        logits = logits[1:seq_len - 2 + 1]  # slice between [CLS] and [SEP] to get valid logits
        logits[logits >= 0.5] = 1
        logits[logits < 0.5] = 0
        offset_mapping = offset_mapping[1:seq_len - 2 + 1]
        predictions = []
        for token in logits:
            predictions.append(np.argwhere(token == 1).tolist())

        # format predictions into example-style output
        text_raw = example.context
        complex_relation_label = [8, 10, 26, 32, 46]
        complex_relation_affi_label = [9, 11, 27, 28, 29, 33, 47]

        # flatten predictions then retrival all valid subject id
        flatten_predictions = []
        for layer_1 in predictions:
            for layer_2 in layer_1:
                flatten_predictions.append(layer_2[0])
        subject_id_list = []
        for cls_label in list(set(flatten_predictions)):
            if 1 < cls_label <= 56 and (cls_label + 55) in flatten_predictions:
                subject_id_list.append(cls_label)
        subject_id_list = list(set(subject_id_list))

        # fetch all valid spo by subject id
        entity_list = []
        spo_list = []
        for id_ in subject_id_list:
            if id_ in complex_relation_affi_label:
                continue  # do this in the next "else" branch
            if id_ not in complex_relation_label:
                subjects = find_entity(text_raw,
                                       id_,
                                       predictions,
                                       offset_mapping)
                objects = find_entity(text_raw,
                                      id_ + 55,
                                      predictions,
                                      offset_mapping)
                for subject_ in subjects:
                    for object_ in objects:
                        spo_list.append({
                            "predicate": id2spo['predicate'][id_],
                            "object": {'@value': object_},
                            "object_type": {'@value': id2spo['object_type'][id_]},
                            "subject": subject_,
                            'subject_type': id2spo['subject_type'][id_]
                        })
            else:
                #  traverse all complex relation and look through their corresponding affiliated objects
                subjects = find_entity(text_raw,
                                       id_,
                                       predictions,
                                       offset_mapping)
                objects = find_entity(text_raw,
                                      id_ + 55,
                                      predictions,
                                      offset_mapping)
                for subject_ in subjects:
                    for object_ in objects:
                        object_dict = {'@value': object_}
                        object_type_dict = {'@value': id2spo['object_type'][id_].split('_')[0]}
                        if id_ in [8, 10, 32, 46] and id_ + 1 in subject_id_list:
                            id_affi = id_ + 1
                            object_dict[id2spo['object_type'][id_affi].split(
                                '_')[1]] = find_entity(text_raw,
                                                       id_affi + 55,
                                                       predictions,
                                                       offset_mapping)[0]
                            object_type_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                id2spo['object_type'][id_affi].split('_')[0]
                        elif id_ == 26:
                            for id_affi in [27, 28, 29]:
                                if id_affi in subject_id_list:
                                    object_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                        find_entity(text_raw,
                                                    id_affi + 55,
                                                    predictions,
                                                    offset_mapping)[0]
                                    object_type_dict[id2spo['object_type'][id_affi].split('_')[1]] = \
                                        id2spo['object_type'][id_affi].split('_')[0]
                        spo_list.append({
                            "predicate": id2spo['predicate'][id_],
                            "object": object_dict,
                            "object_type": object_type_dict,
                            "subject": subject_,
                            "subject_type": id2spo['subject_type'][id_],
                        })
            entity_list.extend(subjects)
            entity_list.extend(objects)
        answer_dict[i][0].extend(entity_list)
        answer_dict[i][1].extend(spo_list)

def write_prediction_results(formatted_outputs, file_path):
    """write the prediction results"""

    with codecs.open(file_path, 'w', 'utf-8') as f:
        for formatted_instance in formatted_outputs:
            json_str = json.dumps(formatted_instance, ensure_ascii=False)
            f.write(json_str)
            f.write('\n')


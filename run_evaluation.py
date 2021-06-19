# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# imitations under the License.
"""
This module to calculate precision, recall and f1-value
of the predicated results.
"""
import numpy as np
from config.mpn.spo_config import SPO_TAG
from utils.utils import logger


def convert_spo_contour(qids, subject_preds, po_preds, eval_file, answer_dict):
    for qid, subject, po_pred in zip(qids.data.cpu().numpy(),
                                     subject_preds.data.cpu().numpy(),
                                     po_preds.data.cpu().numpy()):

        subject = tuple(subject.tolist())

        if qid == -1:
            continue
        spoes = answer_dict[qid][2]
        if subject not in spoes:
            spoes[subject] = []
        tokens = eval_file[qid.item()].bert_tokens
        context = eval_file[qid.item()].context
        tok_to_orig_start_index = eval_file[qid.item()].tok_to_orig_start_index
        tok_to_orig_end_index = eval_file[qid.item()].tok_to_orig_end_index
        start = np.where(po_pred[:, :, 0] > 0.6)
        end = np.where(po_pred[:, :, 1] > 0.5)

        # 留意解码方式，一个主语有可能有多个关系存在
        for _start, predicate1 in zip(*start):
            if _start > len(tokens) - 2 or _start == 0:
                continue
            for _end, predicate2 in zip(*end):
                if _start <= _end <= len(tokens) - 2 and predicate1 == predicate2:
                    spoes[subject].append((_start, _end, predicate1))
        if qid not in answer_dict:
            raise ValueError('error in answer_dict ')
        else:
            answer_dict[qid][0].append(
                context[tok_to_orig_start_index[subject[0] - 1]:tok_to_orig_end_index[subject[1] - 1] + 1])

def convert_spo_contour2(qids, end_list, subjects_str, output_logits, eval_file, answer_dict):
    for qid, output_logit in zip(qids.data.cpu().numpy(),
                                 output_logits):
        if qid not in answer_dict:
            raise ValueError('error in answer_dict ')

        context = eval_file[qid].context

        subjects = subjects_str[qid]
        tok_to_orig_start_index = eval_file[qid].tok_to_orig_start_index
        tok_to_orig_end_index = eval_file[qid].tok_to_orig_end_index
        for subject in subjects:
            answer_dict[qid][0].append(
                context[tok_to_orig_start_index[subject[0] - 1]:tok_to_orig_end_index[subject[1] - 1] + 1])

        spoes = answer_dict[qid][2]

        s_e_o = np.where(output_logit > 0.5)
        for i in range(len(s_e_o[0])):
            s_end = s_e_o[0][i]
            o_end = s_e_o[1][i]
            predicate = s_e_o[2][i]
            if s_end in end_list[qid] and o_end in end_list[qid]:
                s = subjects[end_list[qid].index(s_end)]
                o = subjects[end_list[qid].index(o_end)]
                if s in spoes:
                    spoes[s].append((o[0], o[1], predicate.item()))
                else:
                    spoes[s] = [(o[0], o[1], predicate.item())]

def convert2ressult(args, eval_file, answer_dict):
    for qid in answer_dict.keys():
        spoes = answer_dict[qid][2]
        context = eval_file[qid].context
        tok_to_orig_start_index = eval_file[qid].tok_to_orig_start_index
        tok_to_orig_end_index = eval_file[qid].tok_to_orig_end_index

        complex_relation_label = [6, 8, 24, 30, 44]
        complex_relation_affi_label = [7, 9, 25, 26, 27, 31, 45]

        po_predict = []
        for s, po in spoes.items():
            po.sort(key=lambda x: x[2])
            sub_ent = context[tok_to_orig_start_index[s[0] - 1]:tok_to_orig_end_index[s[1] - 1] + 1].replace('\xa0',
                                                                                                             '')
            for (o1, o2, p) in po:
                object_list = []
                obj_ent = context[tok_to_orig_start_index[o1 - 1]:tok_to_orig_end_index[o2 - 1] + 1].replace('\xa0',
                                                                                                             '')

                object_dict = {'@value': obj_ent}
                # object_list.append('@value' + '[SEP]' + obj_ent)
                object_type_dict = {
                    '@value': SPO_TAG['object_type'][p].split('_')[0]}

                if p in complex_relation_label:
                    predicate = args.id2rel[p].split('_')[0]
                else:
                    predicate = args.id2rel[p]

                if p in complex_relation_affi_label:
                    continue

                def check_object(spoes):

                    for (o1_, o2_) in spoes.keys():
                        obj_ent_ = context[
                                   tok_to_orig_start_index[o1_ - 1]:tok_to_orig_end_index[o2_ - 1] + 1].replace(
                            '\xa0',
                            '')
                        if obj_ent_ == obj_ent:
                            return o1_, o2_
                    return -1, -1

                if p in [6, 8, 30, 44]:
                    candidate_dict = dict()
                    if (o1, o2) not in spoes:
                        o1, o2 = check_object(spoes)

                    if (o1, o2) in spoes:
                        for o1_, o2_, p_ in spoes[(o1, o2)]:
                            if p + 1 == p_:
                                candidate_dict[p_] = (o1_, o2_)

                    for p_, (o1_, o2_) in candidate_dict.items():
                        obj_ent = context[
                                  tok_to_orig_start_index[o1_ - 1]:tok_to_orig_end_index[o2_ - 1] + 1].replace(
                            '\xa0',
                            '')
                        object_dict[args.id2rel[p_].split('_')[1]] = obj_ent
                        object_type_dict[SPO_TAG['object_type'][p_].split('_')[1]] = \
                            SPO_TAG['object_type'][p_].split('_')[0]
                elif p == 24:
                    candidate_dict = dict()

                    if (o1, o2) not in spoes:
                        o1, o2 = check_object(spoes)

                    if (o1, o2) in spoes:
                        for o1_, o2_, p_ in spoes[(o1, o2)]:
                            if p_ in [25, 26, 27]:
                                candidate_dict[p_] = (o1_, o2_)

                    for p_, (o1_, o2_) in candidate_dict.items():

                        if p_ in [25, 26, 27]:
                            obj_ent = context[
                                      tok_to_orig_start_index[o1_ - 1]:tok_to_orig_end_index[o2_ - 1] + 1].replace(
                                '\xa0',
                                '')
                            object_dict[args.id2rel[p_].split('_')[1]] = obj_ent
                            object_type_dict[SPO_TAG['object_type'][p_].split('_')[1]] = \
                                SPO_TAG['object_type'][p_].split('_')[0]

                po_predict.append({
                    "predicate": predicate,
                    "object": object_dict,
                    "object_type": object_type_dict,
                    "subject": sub_ent,
                    "subject_type": SPO_TAG['subject_type'][p]
                })
        answer_dict[qid][1].extend(po_predict)


def run_evaluate(eval_file, answer_dict, chosen):
    entity_em = 0
    entity_pred_num = 0
    entity_gold_num = 0
    tp, fp, fn = 0, 0, 0
    for key in answer_dict.keys():
        triple_gold = eval_file[key].gold_answer
        entity_gold = eval_file[key].sub_entity_list if hasattr(eval_file[key], "sub_entity_list") else eval_file[key].entity_list

        entity_pred = answer_dict[key][0]
        triple_pred = answer_dict[key][1]

        entity_em += len(set(entity_pred) & set(entity_gold))
        entity_pred_num += len(set(entity_pred))
        entity_gold_num += len(set(entity_gold))

        tp_tmp, fp_tmp, fn_tmp = calculate_metric(
            triple_gold, triple_pred)
        tp += tp_tmp
        fp += fp_tmp
        fn += fn_tmp

    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f = 2 * p * r / (p + r) if p + r != 0 else 0

    entity_precision = 100.0 * entity_em / entity_pred_num if entity_pred_num > 0 else 0.
    entity_recall = 100.0 * entity_em / entity_gold_num if entity_gold_num > 0 else 0.
    entity_f1 = 2 * entity_recall * entity_precision / (entity_recall + entity_precision) if (
                                                                                                     entity_recall + entity_precision) != 0 else 0.0

    logger.info('============================================')
    logger.info("{}/entity_em: {},\nentity_pred_num&entity_gold_num: {}\t{} ".format(chosen, entity_em, entity_pred_num,
                                                                                     entity_gold_num))
    logger.info(
        "{}/entity_f1: {}, \nentity_precision: {},\nentity_recall: {} ".format(chosen, entity_f1, entity_precision,
                                                                               entity_recall))
    logger.info('============================================')
    logger.info("{}/em: {},\npre&gold: {}\t{} ".format(chosen, tp, tp + fp, tp + fn))
    logger.info("{}/f1: {}, \nPrecision: {},\nRecall: {} ".format(chosen, f * 100, p * 100,
                                                                  r * 100))
    return {'f1': f, "recall": r, "precision": p}

def calculate_metric(spo_list_gt, spo_list_predict):
    # calculate golden metric precision, recall and f1
    # may be slightly different with final official evaluation on test set,
    # because more comprehensive detail is considered (e.g. alias)
    tp, fp, fn = 0, 0, 0

    for spo in spo_list_predict:
        flag = 0
        for spo_gt in spo_list_gt:
            if spo['predicate'] == spo_gt['predicate'] and spo['object'] == spo_gt['object'] and spo['subject'] == \
                    spo_gt['subject']:
                flag = 1
                tp += 1
                break

        if flag == 0:
            fp += 1
    # if fp != 0:
    #     print(text)
    #     print(spo_list_gt)
    #     print('$' * 10)
    #     print(spo_list_predict)
    #     print()

    '''
    for spo in spo_list_predict:
        if spo in spo_list_gt:
            tp += 1
        else:
            fp += 1
            '''

    fn = len(spo_list_gt) - tp
    return tp, fp, fn

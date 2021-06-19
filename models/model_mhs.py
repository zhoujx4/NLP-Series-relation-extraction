from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertPreTrainedModel


class model_mhs(BertPreTrainedModel):

    def __init__(self, config, E_num, E_em_size, R_num):
        super(model_mhs, self).__init__(config, E_num, E_em_size, R_num)
        self.bert = BertModel(config)
        self.linear_start = nn.Linear(config.hidden_size, E_num + 1)
        self.linear_end = nn.Linear(config.hidden_size, E_num + 1)
        self.E_emb = nn.Embedding(E_num + 1, E_em_size)

        self.linear_head = nn.Linear(config.hidden_size + E_em_size, 128)
        self.linear_base = nn.Linear(config.hidden_size + E_em_size, 128)

        self.rel1 = nn.Linear(128 * 2, 128)
        self.rel2 = nn.Linear(128, R_num)

        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        self.init_weights()

    def forward(self,
                q_ids=None,
                batch_token_ids=None,
                batch_subject_type_ids=None,
                batch_subject_labels=None,
                batch_object_labels=None,
                is_eval=False):
        mask = batch_token_ids != 0
        bert_encoder = self.bert(batch_token_ids, attention_mask=mask)[0]  # [B, L, H]
        seq_len = bert_encoder.size(1)

        if not is_eval:
            start_logits = self.linear_start(bert_encoder)  # [B, L, E_num+1]
            end_logits = self.linear_end(bert_encoder)  # [B, L, E_num+1]

            E_emb = self.E_emb(batch_subject_type_ids)  # [B, L, E_em_size]
            concat_cs = torch.cat([bert_encoder, E_emb], dim=-1)  # [B, L, H+E_em_size]

            # 把 bert的输出 和 实体标签 拼接起来
            f1 = self.linear_head(concat_cs)  # [B, L, 128]
            f2 = self.linear_base(concat_cs)  # [B, L, 128]

            f1 = f1.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [B, L, L, 128]
            f2 = f2.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [B, L, L, 128]
            concat_f = torch.cat([f1, f2], dim=-1)  # [B, L, L, 256]

            output_logits = torch.relu(self.rel1(concat_f))  # [B, L, L, 128]
            output_logits = self.rel2(output_logits)  # [B, L, L, R_num]

            loss_sub = self.loss_fct(start_logits, batch_subject_labels[:, :, 0, :]) * mask.unsqueeze(2) \
                       + self.loss_fct(end_logits, batch_subject_labels[:, :, 1, :]) * mask.unsqueeze(2)
            loss_sub = torch.sum(loss_sub / 2) / torch.sum(mask)
            loss_rel = self.loss_fct(output_logits, batch_object_labels) * (
                (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(-1))
            loss_rel = torch.sum(loss_rel) / torch.sum(mask)
            loss = loss_sub * 10 + loss_rel

            return loss, loss_sub, loss_rel

        else:
            start_logits = torch.sigmoid(self.linear_start(bert_encoder))  # [B, L, E_num+1]
            end_logits = torch.sigmoid(self.linear_end(bert_encoder))  # [B, L, E_num+1]
            start_logits = start_logits * mask.unsqueeze(-1)  # [B, L, E_num+1]
            end_logits = end_logits * mask.unsqueeze(-1)  # [B, L, E_num+1]
            start_logits = start_logits.cpu().numpy()
            end_logits = end_logits.cpu().numpy()

            s_top = torch.zeros(batch_token_ids.size()[:2], dtype=torch.long, device=bert_encoder.device)  # [B, L]
            q_ids = q_ids.cpu().tolist()
            subjects_str = defaultdict(list)
            end_list = defaultdict(list)
            for m in range(len(q_ids)):
                start, start_tp = np.where(start_logits[m] > 0.5)
                end, end_tp = np.where(end_logits[m] > 0.5)
                for i, t in zip(start, start_tp):
                    j = end[end >= i]
                    te = end_tp[end >= i]
                    if len(j) > 0 and te[0] == t:  # TODO 有可能会有实体嵌套的情况？
                        j = j[0]
                        subjects_str[q_ids[m]].append((i, j))
                        end_list[q_ids[m]].append(j)
                        s_top[m][j] = t

            E_emb = self.E_emb(s_top)  # [B, L, E_em_size]
            concat_cs = torch.cat([bert_encoder, E_emb], dim=-1)  # [B, L, H+E_em_size]

            f1 = self.linear_head(concat_cs)  # [B, L, 128]
            f2 = self.linear_base(concat_cs)  # [B, L, 128]

            f1 = f1.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [B, L, L, 128]
            f2 = f2.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [B, L, L, 128]
            concat_f = torch.cat([f1, f2], dim=-1)  # [B, L, L, 256]

            output_logits = torch.relu(self.rel1(concat_f))  # [B, L, L, 128]
            output_logits = torch.sigmoid(self.rel2(output_logits))  # [B, L, L, R_num]
            output_logits = output_logits * ((mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(-1))
            output_logits = output_logits.cpu().numpy()

            return end_list, subjects_str, output_logits


if __name__ == '__main__':
    pass

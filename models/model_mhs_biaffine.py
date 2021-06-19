from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertPreTrainedModel


class Biaffine(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(Biaffine, self).__init__()
        self.w1 = torch.nn.Parameter(nn.init.xavier_uniform_(torch.ones((in_size, out_size, in_size))),
                                     requires_grad=True)
        self.w2 = torch.nn.Parameter(nn.init.xavier_uniform_(torch.ones((2 * in_size + 1, out_size))),
                                     requires_grad=True)

    def forward(self, input1, input2, seq_len):
        f1 = input1.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [B, L, L, 128+128]
        f2 = input1.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [B, L, L, 128+128]
        concat_f1f2 = torch.cat((f1, f2), axis=-1)  # [B, L, L, 256*2]
        concat_f1f2 = torch.cat((concat_f1f2, torch.ones_like(concat_f1f2[..., :1])), axis=-1)  # [B, L, L, 256*2+1]

        # bxi,oij,byj->boxy
        logits_1 = torch.einsum('bxi,ioj,byj->bxyo', input1, self.w1, input2)
        logits_2 = torch.einsum('bijy,yo->bijo', concat_f1f2, self.w2)

        return logits_1 + logits_2  # [B, L, L, R]


class model_mhs_biaffine(BertPreTrainedModel):
    def __init__(self, config, E_num, E_em_size, R_num):
        super(model_mhs_biaffine, self).__init__(config, E_num, E_em_size, R_num)
        self.bert = BertModel(config)
        self.biaffine_layer = Biaffine(128 + E_em_size, R_num)
        self.linear_start1 = nn.Linear(config.hidden_size, 256)
        self.linear_start2 = nn.Linear(256, E_num + 1)
        self.linear_end1 = nn.Linear(config.hidden_size, 256)
        self.linear_end2 = nn.Linear(256, E_num + 1)
        self.cs_emb = nn.Embedding(E_num + 1, E_em_size)

        self.f11 = nn.Linear(config.hidden_size * 2, 256)
        self.f12 = nn.Linear(256, 128)
        self.f21 = nn.Linear(config.hidden_size * 2, 256)
        self.f22 = nn.Linear(256, 128)
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
        bert_output = self.bert(batch_token_ids, attention_mask=mask, output_hidden_states=True)
        seq_len = bert_output[0].size(1)

        if not is_eval:
            layer_1 = bert_output[2][-1]
            layer_2 = bert_output[2][-2]

            start_logits = torch.relu(self.linear_start1(layer_1))  # [B, L, E_num+1]
            start_logits = self.linear_start2(start_logits)  # [B, L, E_num+1]
            end_logits = torch.relu(self.linear_end1(layer_1))  # [B, L, E_num+1]
            end_logits = self.linear_end2(end_logits)  # [B, L, E_num+1]

            cs_emb = self.cs_emb(batch_subject_type_ids)

            concat_cs = torch.cat([layer_1, layer_2], dim=-1)  # [B, L, H*2]

            f1 = torch.relu(self.f11(concat_cs))
            f1 = torch.relu(self.f12(f1))
            f1 = torch.cat([f1, cs_emb], dim=-1)  # [B, L, 128+128]

            f2 = torch.relu(self.f21(concat_cs))
            f2 = torch.relu(self.f22(f2))
            f2 = torch.cat([f2, cs_emb], dim=-1)  # [B, L, 128+128]

            ##
            output_logits = self.biaffine_layer(f1, f2, seq_len)  # [B, L, L, R]

            loss_sub = self.loss_fct(start_logits, batch_subject_labels[:, :, 0, :]) * mask.unsqueeze(2) \
                       + self.loss_fct(end_logits, batch_subject_labels[:, :, 1, :]) * mask.unsqueeze(2)
            loss_sub = torch.sum(loss_sub / 2) / torch.sum(mask)
            loss_rel = self.loss_fct(output_logits, batch_object_labels) * (
                (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(-1))
            loss_rel = torch.sum(loss_rel) / torch.sum(mask)
            loss = loss_sub * 10 + loss_rel

            return loss, loss_sub, loss_rel

        else:
            layer_1 = bert_output[2][-1]
            layer_2 = bert_output[2][-2]

            start_logits = torch.relu(self.linear_start1(layer_1))  # [B, L, E_num+1]
            start_logits = torch.sigmoid(self.linear_start2(start_logits))  # [B, L, E_num+1]
            start_logits = start_logits * mask.unsqueeze(-1)
            start_logits = start_logits.cpu().numpy()
            end_logits = torch.relu(self.linear_end1(layer_1))  # [B, L, E_num+1]
            end_logits = torch.sigmoid(self.linear_end2(end_logits))  # [B, L, E_num+1]
            end_logits = end_logits * mask.unsqueeze(-1)
            end_logits = end_logits.cpu().numpy()

            s_top = torch.zeros(batch_token_ids.size()[:2], dtype=torch.long, device=mask.device)
            q_ids = q_ids.cpu().tolist()
            subjects_str = defaultdict(list)
            end_list = defaultdict(list)
            for m in range(len(q_ids)):

                start = np.where(start_logits[m] > 0.5)[0]
                start_tp = np.where(start_logits[m] > 0.5)[1]
                end = np.where(end_logits[m] > 0.5)[0]
                end_tp = np.where(end_logits[m] > 0.5)[1]
                for i, t in zip(start, start_tp):
                    j = end[end >= i]
                    te = end_tp[end >= i]
                    if len(j) > 0 and te[0] == t:
                        j = j[0]
                        subjects_str[q_ids[m]].append((i, j))
                        end_list[q_ids[m]].append(j)
                        s_top[m][j] = t

            cs_emb = self.cs_emb(s_top)

            concat_cs = torch.cat([layer_1, layer_2], dim=-1)  # [B, L, H*2]

            f1 = torch.relu(self.f11(concat_cs))
            f1 = torch.relu(self.f12(f1))
            f1 = torch.cat([f1, cs_emb], dim=-1)  # [B, L, 128+128]

            f2 = torch.relu(self.f21(concat_cs))
            f2 = torch.relu(self.f22(f2))
            f2 = torch.cat([f2, cs_emb], dim=-1)  # [B, L, 128+128]

            ##
            output_logits = torch.sigmoid(self.biaffine_layer(f1, f2, seq_len))  # [B, L, L, R]
            output_logits = output_logits * ((mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(-1))
            output_logits = output_logits.cpu().numpy()

            return end_list, subjects_str, output_logits


if __name__ == '__main__':
    pass

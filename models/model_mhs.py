import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
from transformers import BertModel
from transformers import BertPreTrainedModel

from utils.utils import batch_gather

class model_mhs(BertPreTrainedModel):
    """
    ERENet : entity relation jointed extraction
    """

    def __init__(self, config, Cs_num):
        super(model_mhs, self).__init__(config, Cs_num, cs_em_size, R_num)
        self.bert = BertModel(config)
        self.linear_start = nn.Linear(config.hidden_size, Cs_num+1)
        self.linear_end = nn.Linear(config.hidden_size, Cs_num+1)
        self.cs_emb = nn.Embedding(Cs_num, cs_em_size)

        self.f1 = nn.Linear(config.hidden_size+cs_em_size, 128)
        self.f2 = nn.Linear(config.hidden_size+cs_em_size, 128)

        self.rel1 = nn.Linear(128*2, 128)
        self.rel2 = nn.Linear(128, R_num)

        self.loss_fct = nn.BCEWithLogitsLoss(reduction='sum')

        self.init_weights()

    def forward(self,
                q_ids=None,
                passage_ids=None,
                segment_ids=None,
                token_type_ids=None,
                cs=None,
                start_tokens=None,
                end_tokens=None,
                c_relation=None,
                eval_file=None,
                is_eval=False):
        mask = (passage_ids != 0).float()
        bert_encoder = self.bert(passage_ids, token_type_ids=segment_ids, attention_mask=mask)[0]
        seq_len = q_ids.size(1)

        if not is_eval:
            start_logits = torch.sigmoid(self.linear_start(bert_encoder))  # [B, L, Cs_num+1]
            end_logits = torch.sigmoid(self.linear_end(bert_encoder))  # [B, L, Cs_num+1]

            cs_emb = self.cs_emb(cs)
            concat_cs = torch.cat([bert_encoder, cs_emb], dim=-1)

            f1 = self.f1(concat_cs)
            f2 = self.f2(concat_cs)

            f1 = f1.unsqueeze(2).expand(-1, -1, seq_len, -1)
            f2 = f2.unsqueeze(1).expand(-1, seq_len, -1, -1)
            concat_f  = torch.cat([f1, f2], dim=-1)

            output_logits = torch.relu(self.rel1(concat_f))
            output_logits = self.rel2(output_logits)  # [B, L, L, R_num]

            loss_sub = self.loss_fct(start_logits, start_tokens) + self.loss_fct(end_logits, end_tokens)
            loss_rel = self.loss_fct(output_logits, c_relation)
            loss = loss_sub + loss_rel

            return loss

        else:
            start_logits = torch.sigmoid(self.linear_start(bert_encoder))  # [B, L, Cs_num+1]
            end_logits = torch.sigmoid(self.linear_end(bert_encoder))
            s_top = np.zeros(passage_ids.size()[:2])
            for m in range(len(start_logits)):
                start = np.where(start_logits[m] > 0.5)[0]
                start_tp = np.where(start_logits[m] > 0.5)[1]
                end = np.where(end_logits[m] > 0.5)[0]
                end_tp = np.where(end_logits[m] > 0.5)[1]
                end_list = []
                for i,t in zip(start,start_tp):
                    j = end[end >= i]
                    te = end_tp[end >= i]
                    if len(j) > 0 and te[0] == t:
                        j = j[0]
                        end_list.append(j)
                        s_top[m][j] = t

            cs_emb = self.cs_emb(s_top)
            concat_cs = torch.cat([bert_encoder, cs_emb], dim=-1)

            f1 = self.f1(concat_cs)
            f2 = self.f2(concat_cs)

            f1 = f1.unsqueeze(2).expand(-1, -1, seq_len, -1)
            f2 = f2.unsqueeze(1).expand(-1, seq_len, -1, -1)
            concat_f  = torch.cat([f1, f2], dim=-1)

            output_logits = torch.relu(self.rel1(concat_f))
            output_logits = self.rel2(output_logits)

            s_e_o = np.where(output_logits[0] > 0.5)
            for i in range(len(s_e_o[0])):
                s_end = s_e_o[0][i]
                o_end = s_e_o[1][i]
                predicate = s_e_o[2][i]
                if s_end in end_list and o_end in end_list:
                    s = subjects_str[end_list.index(s_end)]
                    o = subjects_str[end_list.index(o_end)]
                    p = self.id2p[predicate]
            return

if __name__ == '__main__':
    pass
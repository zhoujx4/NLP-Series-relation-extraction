import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
from transformers import BertModel
from transformers import BertPreTrainedModel

from dataset.data_utils import batch_gather


class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        # >>> m = nn.Linear(20, 30)
        # >>> input = torch.randn(128, 20)
        # >>> output = m(input)
        # >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(ConditionalLayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        self.beta_dense = Linear(hidden_size * 2, hidden_size, bias=False)
        self.gamma_dense = Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, x, cond):
        cond = cond.unsqueeze(1)
        beta = self.beta_dense(cond)
        gamma = self.gamma_dense(cond)
        weight = self.weight + gamma
        bias = self.bias + beta

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return weight * x + bias


class model_mpn(BertPreTrainedModel):
    """
    ERENet : entity relation jointed extraction
    """

    def __init__(self, config, classes_num):
        super(model_mpn, self).__init__(config, classes_num)
        self.classes_num = classes_num

        # BERT model
        self.bert = BertModel(config)
        self.LayerNorm = ConditionalLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # pointer net work
        self.po_dense = nn.Linear(config.hidden_size, self.classes_num * 2)
        self.subject_dense = nn.Linear(config.hidden_size, 2)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        self.init_weights()

    def forward(self,
                q_ids=None,
                passage_ids=None,
                subject_ids=None,
                subject_labels=None,
                object_labels=None,
                eval_file=None,
                is_eval=False):
        mask = (passage_ids != 0).float()
        bert_encoder = self.bert(passage_ids, attention_mask=mask)[0]
        if not is_eval:
            sub_start_encoder = batch_gather(bert_encoder, subject_ids[:, 0])  # [B, H]
            sub_end_encoder = batch_gather(bert_encoder, subject_ids[:, 1])  # [B, H]
            subject = torch.cat([sub_start_encoder, sub_end_encoder], 1)  # [B, H*2]
            context_encoder = self.LayerNorm(bert_encoder, subject)  # [B, L, H]

            sub_preds = self.subject_dense(bert_encoder)  # [B, L, 2]
            po_preds = self.po_dense(context_encoder)  # [B, L, R*2]
            po_preds = po_preds.reshape(passage_ids.size(0), -1, self.classes_num, 2)  # [B, L, R, 2]

            subject_loss = self.loss_fct(sub_preds, subject_labels)  # [B, L, 2]
            # subject_loss = F.binary_cross_entropy(torch.sigmoid(sub_preds) ** 2, subject_labels, reduction='none')
            subject_loss = subject_loss.mean(2)  # [B, L]
            subject_loss = torch.sum(subject_loss * mask.float()) / torch.sum(mask.float())

            po_loss = self.loss_fct(po_preds, object_labels)  # [B, L, R, 2]
            # po_loss = F.binary_cross_entropy(torch.sigmoid(po_preds) ** 2, object_labels, reduction='none')  # [B, L, R, 2]
            po_loss = torch.sum(po_loss.mean(3), 2)  # [B, L]
            po_loss = torch.sum(po_loss * mask.float()) / torch.sum(mask.float())

            loss = subject_loss + po_loss

            return loss, subject_loss, po_loss

        else:
            subject_preds = torch.sigmoid(self.subject_dense(bert_encoder))  # [B, L, 2]
            subject_preds = subject_preds * mask.unsqueeze(-1)
            answer_list = list()
            for qid, sub_pred in zip(q_ids.cpu().numpy(),
                                     subject_preds.cpu().numpy()):
                context = eval_file[qid].bert_tokens
                start = np.where(sub_pred[:, 0] > 0.6)[0]
                end = np.where(sub_pred[:, 1] > 0.5)[0]
                subjects = []

                for i in start:
                    j = end[end >= i]
                    if i == 0 or i > len(context) - 2:
                        continue

                    if len(j) > 0:
                        j = j[0]
                        if j > len(context) - 2:
                            continue
                        subjects.append((i, j))

                answer_list.append(subjects)

            qid_ids, pass_ids, subject_ids, bert_encoders = [], [], [], []
            for i, subjects in enumerate(answer_list):
                if subjects:
                    qid = q_ids[i].unsqueeze(0).expand(len(subjects))  # [len(s), 1]
                    pass_tensor = passage_ids[i, :].unsqueeze(0).expand(len(subjects),
                                                                        passage_ids.size(1))  # [len(s), L]
                    new_bert_encoder = bert_encoder[i, :, :].unsqueeze(0).expand(len(subjects),
                                                                                 bert_encoder.size(1),
                                                                                 bert_encoder.size(2))  # [len(s), L, H]

                    qid_ids.append(qid)
                    pass_ids.append(pass_tensor)
                    subject_ids.append(torch.tensor(subjects, dtype=torch.long))
                    bert_encoders.append(new_bert_encoder)

            if len(qid_ids) == 0:
                subject_ids = torch.zeros(1, 2).long().to(bert_encoder.device)  # [1, 2]
                qid_tensor = torch.tensor([-1], dtype=torch.long).to(bert_encoder.device)  # [1]
                po_tensor = torch.zeros(1, bert_encoder.size(1)).long().to(bert_encoder.device)  # [1, L]
                return qid_tensor, subject_ids, po_tensor

            qids = torch.cat(qid_ids).to(bert_encoder.device)  # [sum(len(x))), 1]
            pass_ids = torch.cat(pass_ids).to(bert_encoder.device)  # [sum(len(x)), L]
            bert_encoders = torch.cat(bert_encoders).to(bert_encoder.device)  # [sum(len(x)), L, H]
            subject_ids = torch.cat(subject_ids).to(bert_encoder.device)  # [sum(len(x)), 2]

            flag = False
            split_heads = 1024

            bert_encoders_ = torch.split(bert_encoders, split_heads, dim=0)
            pass_ids_ = torch.split(pass_ids, split_heads, dim=0)
            subject_encoder_ = torch.split(subject_ids, split_heads, dim=0)

            po_preds = list()
            for i in range(len(bert_encoders_)):
                bert_encoders = bert_encoders_[i]
                pass_ids = pass_ids_[i]
                subject_encoder = subject_encoder_[i]

                if bert_encoders.size(0) == 1:
                    flag = True
                    # print('flag = True**********')
                    bert_encoders = bert_encoders.expand(2, bert_encoders.size(1), bert_encoders.size(2))
                    subject_encoder = subject_encoder.expand(2, subject_encoder.size(1))

                sub_start_encoder = batch_gather(bert_encoders, subject_encoder[:, 0])  # [B, H]
                sub_end_encoder = batch_gather(bert_encoders, subject_encoder[:, 1])  # [B, H]
                subject = torch.cat([sub_start_encoder, sub_end_encoder], 1)  # [B, H*2]
                context_encoder = self.LayerNorm(bert_encoders, subject)  # # [B, L, H]

                po_pred = self.po_dense(context_encoder)  # [B, L, R*2]
                po_pred = po_pred.reshape(pass_ids.size(0), -1, self.classes_num, 2)  # [B, L, R, 2]

                if flag:
                    po_pred = po_pred[1, :, :, :].unsqueeze(0)

                po_preds.append(po_pred)

            po_tensor = torch.cat(po_preds).to(qids.device)  # [sum((len(x)), L, R, 2]
            po_tensor = nn.Sigmoid()(po_tensor)  # [sum((len(x)), B, L, R, 2]
            return qids, subject_ids, po_tensor


if __name__ == '__main__':
    pass

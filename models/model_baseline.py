import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class model_baseline(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(model_baseline, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 250)
        self.fc2 = nn.Linear(250, num_classes)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None):
        mask = input_ids != 0
        output = self.bert(input_ids,
                           attention_mask=mask)
        sequence_output, pooled_output = output[0], output[1]
        output1 = F.relu(self.fc1(sequence_output))
        logits = self.fc2(output1)

        return logits

if __name__ == '__main__':
    model = model_baseline("/data/zhoujx/prev_trained_model/rbt3", num_classes=10)

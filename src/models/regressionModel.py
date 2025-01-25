import torch
import torch.nn as nn
from transformers import BertModel, AutoModel

class RegressionModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        # load pretrained model
        self.bert = AutoModel.from_pretrained(model_name)
        # add regression layer
        self.regression_layer = nn.Linear(self.bert.config.hidden_size, 1) # 768 x 1 -> output layer regression

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        score = self.regression_layer(cls_embedding)
        return score
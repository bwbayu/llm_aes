import torch
import torch.nn as nn
from transformers import AutoModel

class RegressionModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', dropout=0.1, freeze_transformer=False):
        super().__init__()
        # load pretrained model
        self.model = AutoModel.from_pretrained(model_name)

        # Freeze transformer layers if needed
        if freeze_transformer:
            for param in self.model.parameters():
                param.requires_grad = False  

        # add regression layer
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.regression_layer = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_embedding)
        score = self.regression_layer(x)
        return score
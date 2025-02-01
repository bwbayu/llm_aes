import torch
from torch import nn
from transformers import LongformerPreTrainedModel, LongformerModel

class LongformerForRegression(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.longformer = LongformerModel(config)
        self.regression_head = nn.Linear(config.hidden_size, 1)  # Output a single value
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
            output_attentions=False, output_hidden_states=False, return_dict=True):

        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            inputs_embeds=inputs_embeds, 
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        pooled_output = outputs.last_hidden_state[:, 0]  # Take CLS token output
        logits = self.regression_head(pooled_output)  # Regression output
        
        return logits
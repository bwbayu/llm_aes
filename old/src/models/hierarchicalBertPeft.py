import torch
import torch.nn as nn
from transformers import AutoModel

class HierarchicalBertPeft(nn.Module):
    def __init__(self, model_name='bert-base-uncased', *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load pretrained model
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        # add regression layer
        self.regression_layer = nn.Linear(self.config.hidden_size, 1) # 768 x 1

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, inputs_embeds=None, labels=None,
                output_attentions=False, output_hidden_states=False, return_dict=True):
        num_batch, num_chunks, seq_len = input_ids.shape
        batch_outputs = []
        for idx_batch in range(num_batch):
            chunk_outputs = []
            for idx_chunk in range(num_chunks):
                chunk_ids = input_ids[idx_batch, idx_chunk, :].to(self.bert.device)
                if chunk_ids[0] == 0:
                    continue 
                chunk_mask = attention_mask[idx_batch, idx_chunk, :].to(self.bert.device)
                chunk_type = token_type_ids[idx_batch, idx_chunk, :].to(self.bert.device)
                outputs = self.bert(
                    input_ids=chunk_ids.unsqueeze(0),
                    attention_mask=chunk_mask.unsqueeze(0),
                    token_type_ids=chunk_type.unsqueeze(0),
                    inputs_embeds=inputs_embeds, 
                    output_attentions=output_attentions, 
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
                chunk_outputs.append(outputs.last_hidden_state[:, 0, :])
            
            combined_output = self.attention_pooling(chunk_outputs, num_chunks)
            batch_outputs.append(combined_output)

        batch_outputs = torch.stack(batch_outputs)
        logits = self.regression_layer(batch_outputs)
        return logits

    def attention_pooling(self, chunk_outputs, num_chunks):
        """Menggabungkan output chunks menggunakan attention pooling."""
        if num_chunks == 1:
            # Skip attention pooling and return the single chunk
            return chunk_outputs[0].squeeze(0)
        stacked_chunks = torch.cat(chunk_outputs, dim=0)  # [num_chunks, hidden_size]
        attention_weights = torch.nn.functional.softmax(stacked_chunks @ stacked_chunks.T, dim=1)
        pooled_output = attention_weights @ stacked_chunks
        return pooled_output.mean(dim=0)  # [hidden_size]

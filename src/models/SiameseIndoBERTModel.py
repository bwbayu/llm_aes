import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, BertTokenizer

class SiameseIndoBERTModel(nn.Module):
    def __init__(self, model_name='indobenchmark/indobert-lite-base-p2'):
        super(SiameseIndoBERTModel, self).__init__()
        # Load the IndoBERT model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.bert_model.config.hidden_size
        
        # Add pooling layer
        self.pooling_mode_mean_tokens = True
        self.pooling_mode_cls_token = False
        self.pooling_mode_max_tokens = False
        
        # Add normalization layer
        self.normalize_embeddings = True
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        # Regression head
        self.regression_head = nn.Linear(self.embedding_dim * 2, 1)
        
    def get_embeddings(self, texts):
        # Tokenize the input texts
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        )
        
        # Move to the same device as the model
        encoded_input = {k: v.to(next(self.parameters()).device) for k, v in encoded_input.items()}
        
        # Get attention mask
        attention_mask = encoded_input['attention_mask']
        
        # Get BERT outputs
        outputs = self.bert_model(**encoded_input)
        
        # Get token embeddings
        token_embeddings = outputs.last_hidden_state
        
        # Apply pooling
        token_vecs = []
        
        # Apply mean pooling
        if self.pooling_mode_mean_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            token_vecs.append(sum_embeddings / sum_mask)
            
        # Apply CLS pooling
        if self.pooling_mode_cls_token:
            token_vecs.append(token_embeddings[:, 0])
            
        # Apply max pooling
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_embeddings = torch.max(token_embeddings, 1)[0]
            token_vecs.append(max_embeddings)
            
        # Concatenate the pooling results if multiple pooling methods are used
        sentence_embeddings = torch.cat(token_vecs, 1) if len(token_vecs) > 1 else token_vecs[0]
        
        # Normalize embeddings if specified
        if self.normalize_embeddings:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
        return sentence_embeddings
        
    def forward(self, reference_texts, student_texts):
        # Get embeddings while maintaining the computation graph
        reference_embeddings = self.get_embeddings(reference_texts)
        student_embeddings = self.get_embeddings(student_texts)
        
        # Concatenate the embeddings
        combined = torch.cat((reference_embeddings, student_embeddings), dim=1)
        
        x = self.dropout(combined)
        # Final regression score
        score = torch.sigmoid(self.regression_head(x))
        
        return score
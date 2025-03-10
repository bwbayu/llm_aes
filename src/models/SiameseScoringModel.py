import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertTokenizer

class SiameseScoringModel(nn.Module):
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', dropout=0.1):
        super(SiameseScoringModel, self).__init__()
        
        # Load the model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get embedding dimension from the model config
        self.embedding_dim = self.encoder.config.hidden_size
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        
        # Regression head
        self.regression_head = nn.Linear(self.embedding_dim * 2, 1)
    
    def mean_pooling(self, model_output, attention_mask):
        # Mean pooling - take average of all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
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
        device = next(self.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Get model output (without torch.no_grad to allow fine-tuning)
        outputs = self.encoder(**encoded_input)
        
        # Apply mean pooling to get sentence embeddings
        embeddings = self.mean_pooling(outputs, encoded_input['attention_mask'])
        return embeddings
    
    def forward(self, reference_texts, student_texts):
        # Get embeddings for reference and student texts
        reference_embeddings = self.get_embeddings(reference_texts)
        student_embeddings = self.get_embeddings(student_texts)
        
        # Concatenate the embeddings
        combined = torch.cat((reference_embeddings, student_embeddings), dim=1)
        
        # Apply dropout
        x = self.dropout(combined)
        
        # Final regression score
        score = self.regression_head(x)
        
        return score
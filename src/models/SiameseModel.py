import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class SiameseModel(nn.Module):
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        super(SiameseModel, self).__init__()
        
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get embedding dimension from the model config
        self.embedding_dim = self.encoder.config.hidden_size
    
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
    
    def forward(self, reference_texts, student_texts, sim_type='cosine'):
        # Get embeddings
        reference_embeddings = self.get_embeddings(reference_texts)
        student_embeddings = self.get_embeddings(student_texts)
        
        # Normalize embeddings
        ref_embedding = F.normalize(reference_embeddings, p=2, dim=1)
        student_embedding = F.normalize(student_embeddings, p=2, dim=1)
        
        if sim_type == 'cosine':
            # Compute cosine similarity
            similarity = torch.sum(ref_embedding * student_embedding, dim=1).unsqueeze(1)
        elif sim_type == 'manhattan':
            # Manhattan distance similarity
            manhattan_distance = torch.sum(torch.abs(ref_embedding - student_embedding), dim=1)
            similarity = (1 / (1 + manhattan_distance)).unsqueeze(1)
        elif sim_type == 'euclidean':
            # Euclidean distance similarity
            euclidean_distance = torch.sqrt(torch.sum((ref_embedding - student_embedding) ** 2, dim=1))
            similarity = (1 / (1 + euclidean_distance)).unsqueeze(1)
        
        return similarity
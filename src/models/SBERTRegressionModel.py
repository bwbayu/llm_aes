import torch
from torch import nn
from sentence_transformers import SentenceTransformer

class SBERTRegressionModel(nn.Module):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        super(SBERTRegressionModel, self).__init__()
        # Load the sentence transformer model
        self.sentence_transformer = SentenceTransformer(model_name)
        self.embedding_dim = self.sentence_transformer.get_sentence_embedding_dimension()
        
        # dropout layer
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        # Regression head
        self.regression_head = nn.Linear(self.embedding_dim * 2, 1)

        
    def forward(self, reference_texts, student_texts):
        reference_features = self.sentence_transformer.tokenize(reference_texts)
        student_features = self.sentence_transformer.tokenize(student_texts)

        if next(self.parameters()).device != reference_features['input_ids'].device:
            reference_features = {k: v.to(next(self.parameters()).device) for k, v in reference_features.items()}
            student_features = {k: v.to(next(self.parameters()).device) for k, v in student_features.items()}
        
        # Get embeddings while maintaining the computation graph
        reference_embeddings = self.sentence_transformer.forward(reference_features)['sentence_embedding']
        student_embeddings = self.sentence_transformer.forward(student_features)['sentence_embedding']
        
        # Concatenate the embeddings
        combined = torch.cat((reference_embeddings, student_embeddings), dim=1)
        
        x = self.dropout(combined)
        # Final regression score
        score = torch.sigmoid(self.regression_head(x))
        
        return score
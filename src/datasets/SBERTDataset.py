import torch
from torch.utils.data import Dataset
import numpy as np
import regex as re

class SBERTDataset(Dataset):
    def __init__(self, dataframe, use_normalized_score=True):
        # convert to list
        self.reference_answers = dataframe['reference_answer'].tolist()
        self.student_answers = dataframe['answer'].tolist() 
        
        # Use either normalized or raw score based on parameter
        if use_normalized_score:
            self.scores = dataframe['normalized_score'].values.astype(np.float32)
        else:
            self.scores = dataframe['score'].values.astype(np.float32)
    
    def preprocess_text(self, text):
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Convert to lowercase
        text = text.lower()
        # Remove special characters (keep punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        return text
    
    def __len__(self):
        return len(self.scores)
    
    def __getitem__(self, idx):
        return {
            'reference_answer': self.preprocess_text(self.reference_answers[idx]),
            'student_answer': self.preprocess_text(self.student_answers[idx]),
            'score': torch.tensor(self.scores[idx], dtype=torch.float)
        }
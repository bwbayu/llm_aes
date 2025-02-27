import torch
from torch.utils.data import Dataset
import re

class AutomaticScoringDataset(Dataset):
    def __init__(self, dataframe, tokenizer, use_reference=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.use_reference = use_reference  # Menentukan apakah menggunakan reference answer atau tidak

    def __len__(self):
        return len(self.dataframe)

    def preprocess_text(self, text):
        text = ' '.join(text.split())  # Hapus spasi berlebih
        text = text.lower()  # Ubah ke lowercase
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # Hapus karakter khusus
        return text

    def __getitem__(self, index):
        student_answer = self.preprocess_text(str(self.dataframe.iloc[index]['answer']))
        score = self.dataframe.iloc[index]['normalized_score']

        if self.use_reference and 'reference_answer' in self.dataframe.columns:
            reference_answer = self.preprocess_text(str(self.dataframe.iloc[index]['reference_answer']))
            encoding = self.tokenizer.encode_plus(
                reference_answer,
                student_answer,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            encoding = self.tokenizer.encode_plus(
                student_answer,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        encoding = {key: tensor.squeeze(0) for key, tensor in encoding.items()}
        encoding['labels'] = torch.tensor(score, dtype=torch.float)

        return encoding

    def get_max_length(self, index):
        """Menghitung panjang maksimum tokenized input berdasarkan apakah reference_answer digunakan atau tidak."""
        student_answer = str(self.dataframe.iloc[index]['answer'])

        if self.use_reference and 'reference_answer' in self.dataframe.columns:
            reference_answer = str(self.dataframe.iloc[index]['reference_answer'])
            encoding = self.tokenizer.encode_plus(
                reference_answer,
                student_answer,
                add_special_tokens=True,
                return_tensors='pt'
            )
        else:
            encoding = self.tokenizer.encode_plus(
                student_answer,
                add_special_tokens=True,
                return_tensors='pt'
            )

        return encoding['input_ids'].flatten().shape[0]

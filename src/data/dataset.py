from torch.utils.data import Dataset
import torch

class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, type):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.type = type

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        question = str(self.df.iloc[index]['question'])
        reference_answer = str(self.df.iloc[index]['reference_answer'])
        student_answer = str(self.df.iloc[index]['answer'])
        score = self.df.iloc[index]['normalized_score2']

        # concat input text
        question = question if question is not None else "[NO_QUESTION]" # handle some dataset that doesn't have question
        text = f"{self.tokenizer.cls_token} Question: {question} Reference Answer: {reference_answer} {self.tokenizer.sep_token} Student Answer: {student_answer} {self.tokenizer.sep_token}"

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens = False, # special token already added manually (default:True)
            max_length = self.max_len, # length of return embedding 
            padding='max_length',  # add padding until value of max_length
            truncation=True, # truncation if text exceed value of max_length
            return_attention_mask = True, # attention mask is used to differentiate between "real" token and padding token (default:True)
            return_tensors = 'pt'
        )

        # create token_type_ids manually, this token is used to differentiate between segment
        token_type_ids = []
        current_token = 0
        flag = 0
        for token in encoding['input_ids'].flatten():
            if(((self.type == "BERT" or self.type == "ALBERT") and token == 0) or (self.type == "LONGFORMER" and token == 1)):
                token_type_ids.append(0)
                continue
            token_type_ids.append(current_token)
            if((self.type == "BERT" and token == 102) or (self.type == "ALBERT" and token == 3) or (self.type == "LONGFORMER" and token == 2) and flag == 0): 
                current_token += 1
                flag = 1
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'scores': torch.tensor(score, dtype=torch.float),
            'token_type_ids': torch.tensor(token_type_ids)
        }
    
    def get_max_length(self, index):
        question = str(self.df.iloc[index]['question'])
        reference_answer = str(self.df.iloc[index]['reference_answer'])
        student_answer = str(self.df.iloc[index]['answer'])

        # concat input text
        question = question if question is not None else "[NO_QUESTION]" # handle some dataset that doesn't have question
        text = f"Question: {question} Reference Answer: {reference_answer} {self.tokenizer.sep_token} Student Answer: {student_answer}"
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
        )

        return encoding['input_ids'].flatten().shape[0]
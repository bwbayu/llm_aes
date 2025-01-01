from torch.utils.data import Dataset
import torch

class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        question = str(self.df.iloc[index]['question'])
        reference_answer = str(self.df.iloc[index]['reference_answer'])
        student_answer = str(self.df.iloc[index]['answer'])
        score = self.df.iloc[index]['normalized_score']

        # concat input text
        question = question if question is not None else "" # handle some dataset that doesn't have question
        text = f"[CLS] Question: {question} Reference Answer: {reference_answer} [SEP] Student Answer: {student_answer} [SEP]"

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
        for token in encoding['input_ids'].flatten():
            if(token == 0):
                token_type_ids.append(0)
                continue
            token_type_ids.append(current_token)
            if(token == 102 or token == 3): # 102 is token SEP for bert-base and 3 is for albert-lite
                current_token += 1
        
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
        question = question if question is not None else "" # handle some dataset that doesn't have question
        text = f"Question: {question} [SEP] Reference Answer: {reference_answer} [SEP] Student Answer: {student_answer}"
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
        )

        return encoding['input_ids'].flatten().shape[0]
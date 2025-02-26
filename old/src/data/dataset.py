from torch.utils.data import Dataset
import torch

class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        # # get special token
        # self.sep_token = tokenizer.encode_plus(tokenizer.sep_token, add_special_tokens=False)['input_ids'][0]
        # self.pad_token = tokenizer.encode_plus(tokenizer.pad_token, add_special_tokens=False)['input_ids'][0]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        question = str(self.df.iloc[index].get('question', "[UNK]"))
        reference_answer = str(self.df.iloc[index]['reference_answer'])
        student_answer = str(self.df.iloc[index]['answer'])
        score = self.df.iloc[index]['normalized_score']

        # concat input text
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

        ''' LongFormer doesn't need token_type_ids because it's base architecture '''
        # # create token_type_ids manually, this token is used to differentiate between segment
        # token_type_ids = []
        # current_token = 0
        # flag = 0
        # for token in encoding['input_ids'].flatten():
        #     if(token == self.pad_token):
        #         token_type_ids.append(0)
        #         continue
        #     token_type_ids.append(current_token)
        #     if((token == self.sep_token) and (flag == 0)):
        #         current_token += 1
        #         flag = 1
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            # 'token_type_ids': torch.tensor(token_type_ids),
        }, torch.tensor(score, dtype=torch.float)
    
    def get_max_length(self, index):
        question = str(self.df.iloc[index]['question'])
        reference_answer = str(self.df.iloc[index]['reference_answer'])
        student_answer = str(self.df.iloc[index]['answer'])

        # concat input text
        question = question if question is not None else "[UNK]" # handle some dataset that doesn't have question
        text = f"Question: {question} Reference Answer: {reference_answer} {self.tokenizer.sep_token} Student Answer: {student_answer}"
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
        )

        return encoding['input_ids'].flatten().shape[0]
    
    def get_each_length(self, index):
        question = str(self.df.iloc[index]['question'])
        reference_answer = str(self.df.iloc[index]['reference_answer'])
        student_answer = str(self.df.iloc[index]['answer'])

        # concat input text
        question = f"Question: {question if question is not None else '[UNK]'}"
        reference = f" Reference Answer: {reference_answer} {self.tokenizer.sep_token}"
        student = f"Student Answer: {student_answer}"
        question_encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            return_tensors='pt',
        )
        reference_encoding = self.tokenizer.encode_plus(
            reference,
            add_special_tokens=True,
            return_tensors='pt',
        )
        student_encoding = self.tokenizer.encode_plus(
            student,
            add_special_tokens=True,
            return_tensors='pt',
        )

        return question_encoding['input_ids'].flatten().shape[0], reference_encoding['input_ids'].flatten().shape[0], student_encoding['input_ids'].flatten().shape[0]
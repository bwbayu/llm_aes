from torch.utils.data import Dataset
import torch

class EssayDataset(Dataset):
    def __init__(self, questions, reference_answers, student_answers, scores, tokenizer, max_len):
        self.questions = questions
        self.reference_answers = reference_answers
        self.student_answers = student_answers
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.student_answers)
    
    def __getitem__(self, index):
        question = str(self.questions[index])
        reference_answer = str(self.reference_answers[index])
        student_answer = str(self.student_answers[index])
        score = self.scores[index]

        # concat input text
        question = question if question is not None else "" # handle some dataset that doesn't have question
        text = f"[CLS] Question: {question} [SEP] Reference Answer: {reference_answer} [SEP] Student Answer: {student_answer} [SEP]"

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens = False, # special token already added manually (default:True)
            max_length = self.max_len, # length of return embedding 
            padding='max_length',  # add padding until value of max_length
            truncation=True, # truncation if text exceed value of max_length
            return_attention_mask = True, # attention mask is used to differentiate between "real" token and padding token (default:True)
            return_tensors = 'pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'scores': torch.tensor(score, dtype=torch.float)
        }
    
    def get_token_type_ids(input_ids):
        # create token_type_ids manually, this token is used to differentiate between segment
        token_type_ids = []
        current_token = 0
        for token in input_ids:
            if(token == 0):
                token_type_ids.append(0)
                continue
            token_type_ids.append(current_token)
            if(token == 102):
                current_token += 1
        
        return torch.tensor(token_type_ids)
    
    def get_max_length(self, index):
        print(index)
        question = str(self.questions[index])
        reference_answer = str(self.reference_answers[index])
        student_answer = str(self.student_answers[index])

        # concat input text
        question = question if question is not None else "" # handle some dataset that doesn't have question
        text = f"Question: {question} [SEP] Reference Answer: {reference_answer} [SEP] Student Answer: {student_answer}"
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
        )

        return encoding['input_ids'].flatten().shape[0]
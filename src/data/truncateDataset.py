from torch.utils.data import Dataset
import torch

class TruncateDataset:
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        question = str(self.df.iloc[index].get('question', "[UNK]"))
        reference_answer = str(self.df.iloc[index]['reference_answer'])
        student_answer = str(self.df.iloc[index]['answer'])
        score = self.df.iloc[index]['normalized_score']

        # Tokenize separately without truncation
        question_tokens = self.tokenizer.tokenize(question)
        ref_tokens = self.tokenizer.tokenize(reference_answer)
        stud_tokens = self.tokenizer.tokenize(student_answer)

        # Special tokens [CLS] and [SEP]
        cls_token = [self.tokenizer.cls_token]  # [CLS]
        sep_token = [self.tokenizer.sep_token]  # [SEP]
        pad_token_id = self.tokenizer.pad_token_id

        # Ensure the question is not truncated
        q_length = len(question_tokens)
        remaining_length = self.max_len - (q_length + 3)  # 3 = [CLS] + [SEP] + [SEP]

        # Initial equal split
        half_remaining = remaining_length // 2
        ref_alloc = min(len(ref_tokens), half_remaining)
        stud_alloc = min(len(stud_tokens), half_remaining)

        # If extra space is left, give it to the longer one
        extra_space = remaining_length - (ref_alloc + stud_alloc)
        if len(ref_tokens) > ref_alloc:
            ref_alloc += extra_space
        elif len(stud_tokens) > stud_alloc:
            stud_alloc += extra_space

        # Truncate
        truncated_ref = ref_tokens[:ref_alloc]
        truncated_stud = stud_tokens[:stud_alloc]

        final_tokens = cls_token + question_tokens + truncated_ref + sep_token + truncated_stud + sep_token

        # Convert to input IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(final_tokens)
        attention_mask = [1] * len(input_ids)

        # Apply padding if the length is less than max_length
        pad_length = self.max_len - len(input_ids)
        if pad_length > 0:
            input_ids += [pad_token_id] * pad_length
            attention_mask += [0] * pad_length

        sep_token = self.tokenizer.encode_plus(self.tokenizer.sep_token, add_special_tokens=False)['input_ids'][0]
        pad_token = self.tokenizer.encode_plus(self.tokenizer.pad_token, add_special_tokens=False)['input_ids'][0]

        # create token_type_ids manually, this token is used to differentiate between segment
        token_type_ids = []
        current_token = 0
        flag = 0
        for token in input_ids:
            if(token == pad_token):
                token_type_ids.append(pad_token_id)
                continue
            token_type_ids.append(current_token)
            if((token == sep_token) and (flag == 0)):
                current_token += 1
                flag = 1
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }, torch.tensor(score, dtype=torch.float)
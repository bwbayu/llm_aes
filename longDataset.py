from torch.utils.data import Dataset
import torch

class LongEssayDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, overlapping):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlapping = overlapping

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        question = str(self.df.iloc[index].get('question', "[NO_QUESTION]"))
        reference_answer = str(self.df.iloc[index]['reference_answer'])
        student_answer = str(self.df.iloc[index]['answer'])
        score = self.df.iloc[index]['normalized_score']

        if(self.df.iloc[index]['max_length1'] > (self.max_len-2)):
            # separate 2 segment for input text
            text1 = f"Question: {question} Reference Answer: {reference_answer}"
            text2 = f"Student Answer: {student_answer}"

            # tokenizer each segment
            tokens1 = self.tokenizer.encode_plus(
                text1, 
                add_special_tokens=False, 
                truncation=False, 
                return_tensors='pt'
            )
            tokens2 = self.tokenizer.encode_plus(
                text2, 
                add_special_tokens=False, 
                truncation=False, 
                return_tensors='pt'
            )

            # create chunk for each segment
            chunks_segment1 = self.create_chunks(tokens1, segment_num=0)
            chunks_segment2 = self.create_chunks(tokens2, segment_num=1)
            chunks = []
            chunks = chunks_segment1 + chunks_segment2

            return chunks, torch.tensor(score, dtype=torch.float)
        else:
            # concat input text
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
    
    def create_token_type(self, input_ids, segment_num):
        # create token_type_ids embedding manually
        token_type_ids = []
        for token in input_ids:
            if token == 0:
                token_type_ids.append(0)
                continue
            token_type_ids.append(segment_num)
        return torch.tensor(token_type_ids)
    
    def create_chunks(self, tokens, segment_num):
        input_ids, attention_mask = tokens['input_ids'].flatten(), tokens['attention_mask'].flatten()
        token_type_ids = self.create_token_type(input_ids, segment_num)
        stride=self.max_len-self.overlapping
        chunk = []
        cls_token = torch.tensor([2])
        sep_token = torch.tensor([3])

        for i in range(0, len(input_ids), stride):
            chunk_ids = input_ids[i: i+(self.max_len - 2)]
            chunk_mask = attention_mask[i: i+(self.max_len - 2)]
            chunk_type = token_type_ids[i: i+(self.max_len - 2)]

            # check if this chunk need to be created or not because probably its just the stride token
            if(len(chunk_ids) <= self.overlapping):
                break
            
            # Tambahkan token [CLS] khusus untuk segment dan chunk pertama 
            if(segment_num == 0 and i == 0):
                chunk_ids = torch.cat([cls_token, chunk_ids])
            else:
                # Add token [SEP] di awal chunk seterusnya
                chunk_ids = torch.cat([sep_token, chunk_ids])

            # Tambahkan [SEP] di akhir tiap chunk
            chunk_ids = torch.cat([chunk_ids, sep_token])
            chunk_mask = torch.cat([torch.ones(1, dtype=torch.long), chunk_mask, torch.ones(1, dtype=torch.long)])
            chunk_type = torch.cat([torch.tensor([segment_num]), chunk_type, torch.tensor([segment_num])])

            # menambahkan padding pada chunk terakhir agar memastikan panjang tiap chunk itu sama
            if len(chunk_ids) < (self.max_len):
                padding_length = (self.max_len) - len(chunk_ids)

                # assign padding 0
                chunk_ids = torch.cat([chunk_ids, torch.zeros(padding_length, dtype=torch.long)])
                chunk_mask = torch.cat([chunk_mask, torch.zeros(padding_length, dtype=torch.long)])
                chunk_type = torch.cat([chunk_type, torch.zeros(padding_length, dtype=torch.long)])

            chunk.append({
                'input_ids': chunk_ids,
                'attention_mask': chunk_mask,
                'token_type_ids': chunk_type
            })

        return chunk
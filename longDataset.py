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
        question = str(self.df.iloc[index]['question'])
        reference_answer = str(self.df.iloc[index]['reference_answer'])
        student_answer = str(self.df.iloc[index]['answer'])
        score = self.df.iloc[index]['normalized_score']

        if(self.df.iloc[index]['max_length1'] > (self.max_len-2)):
            # separate 2 segment for input text
            question = question if question is not None else "[NO_QUESTION]" # handle some dataset that doesn't have question
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

            # get token_type_ids for each segment
            # segment 1
            input_ids1 = tokens1['input_ids'].flatten()
            attention_mask1 = tokens1['attention_mask'].flatten()
            token_type_ids1 = self.create_token_type(input_ids1, 0)

            # segment 2
            input_ids2 = tokens2['input_ids'].flatten()
            attention_mask2 = tokens2['attention_mask'].flatten()
            token_type_ids2 = self.create_token_type(input_ids2, 1)
            print("len tensor : ", str(len(input_ids1)+len(input_ids2)))
            # create chunk for each segment
            chunks_segment1 = self.create_chunks(input_ids1, attention_mask1, token_type_ids1, segment_num=0, stride=self.max_len-self.overlapping)
            chunks_segment2 = self.create_chunks(input_ids2, attention_mask2, token_type_ids2, segment_num=1, stride=self.max_len-self.overlapping)
            chunks = []
            chunks = chunks_segment1 + chunks_segment2

            return chunks, torch.tensor(score, dtype=torch.float)
        else:
            # concat input text
            question = question if question is not None else "[NO_QUESTION]" # handle some dataset that doesn't have question
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
    
    def create_chunks(self,input_ids, attention_mask, token_type_ids, segment_num, stride):
        chunk = []
        cls_token = torch.tensor([2])
        sep_token = torch.tensor([3])

        for i in range(0, len(input_ids), stride):
            flag_padding = 0
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
                flag_padding = 1
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

            if flag_padding == 1:
                # stop stride chunk
                break

        return chunk
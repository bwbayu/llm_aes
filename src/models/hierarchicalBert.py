import torch
import torch.nn as nn
from transformers import AutoModel

class HierarchicalBert(nn.Module):
    def __init__(self, model_name='bert-base-uncased', *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load pretrained model
        self.bert = AutoModel.from_pretrained(model_name)
        # add regression layer
        self.regression_layer = nn.Linear(self.bert.config.hidden_size, 1) # 768 x 1

    def forward(self, batch_chunks):
        batch_outputs = []
        # for loop data di dalam batch -> 4 data berisi list of chunk
        for chunks in batch_chunks:
            chunk_outputs = []
            # for loop data tiap chunk -> per 1 data bisa terdiri dari beberapa chunk
            for chunk in chunks:
                # tiap chunk punya 3 key ini
                input_ids = chunk['input_ids'].unsqueeze(0).to(self.bert.device)
                attention_mask = chunk['attention_mask'].unsqueeze(0).to(self.bert.device)
                token_type_ids = chunk['token_type_ids'].unsqueeze(0).to(self.bert.device)
                # print(f"Size of input_ids: {input_ids.size()} | Device of input_ids: {input_ids.device}")
                # print(f"Size of attention_mask: {attention_mask.size()} | Device of attention_mask: {attention_mask.device}")
                # print(f"Size of token_type_ids: {token_type_ids.size()} | Device of token_type_ids: {token_type_ids.device} | Max Token Type ID: {token_type_ids.max()}, Min Token Type ID: {token_type_ids.min()}")

                # masuk ke arsitektur bert -> outputnya [max_seq x [768 token]]
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                """
                ambil token CLS nya saja karena pada token ini berisi "rangkuman" informasi dari keseluruhan token (token cls ini ada di index 0)
                (alt) 511 token lainnya bisa saja digunakan, namun harus masuk ke mean pooling/attention pooling dulu agar ukuran token berubah dari 512x768 ke 1x768
                """
                chunk_outputs.append(outputs.last_hidden_state[:, 0, :])
            
            """
            jika menggunakan cara alternatif maka proses attention pooling/mean pooling terjadi 2x, yaitu untuk tiap chunk dan antar chunk
            jika menggunakan token CLS saja maka proses attention pooling/mean pooling terjadi hanya untuk antar chunk, karena informasi tiap chunk sudah di ambil lewat token CLS saja 
            """
            # Gabungkan chunks dengan attention pooling
            combined_output = self.attention_pooling(chunk_outputs)
            batch_outputs.append(combined_output)

        # Konversi ke tensor dan aplikasikan regression head
        batch_outputs = torch.stack(batch_outputs)  # [batch_size, hidden_size] -> 4x768
        regression_output = self.regression_layer(batch_outputs)  # [batch_size, output_dim] -> 4x1
        return regression_output

    def attention_pooling(self, chunk_outputs):
        """Menggabungkan output chunks menggunakan attention pooling."""
        stacked_chunks = torch.cat(chunk_outputs, dim=0)  # [num_chunks, hidden_size]
        attention_weights = torch.nn.functional.softmax(stacked_chunks @ stacked_chunks.T, dim=1)
        pooled_output = attention_weights @ stacked_chunks
        return pooled_output.mean(dim=0)  # [hidden_size]

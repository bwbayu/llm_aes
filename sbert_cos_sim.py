# ============================================================================= DATASET
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

# ============================================================================= MODEL
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

class SiameseModel(nn.Module):
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        super(SiameseModel, self).__init__()
        
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Get embedding dimension from the model config
        self.embedding_dim = self.encoder.config.hidden_size
    
    def mean_pooling(self, model_output, attention_mask):
        # Mean pooling - take average of all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embeddings(self, texts):
        # Tokenize the input texts
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        )
        
        # Move to the same device as the model
        device = next(self.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Get model output (without torch.no_grad to allow fine-tuning)
        outputs = self.encoder(**encoded_input)
        
        # Apply mean pooling to get sentence embeddings
        embeddings = self.mean_pooling(outputs, encoded_input['attention_mask'])
        return embeddings
    
    def forward(self, reference_texts, student_texts, sim_type='cosine'):
        # Get embeddings
        reference_embeddings = self.get_embeddings(reference_texts)
        student_embeddings = self.get_embeddings(student_texts)
        
        # Normalize embeddings
        ref_embedding = F.normalize(reference_embeddings, p=2, dim=1)
        student_embedding = F.normalize(student_embeddings, p=2, dim=1)
        
        if sim_type == 'cosine':
            # Compute cosine similarity
            similarity = torch.sum(ref_embedding * student_embedding, dim=1).unsqueeze(1)
        elif sim_type == 'manhattan':
            # Manhattan distance similarity
            manhattan_distance = torch.sum(torch.abs(ref_embedding - student_embedding), dim=1)
            similarity = (1 / (1 + manhattan_distance)).unsqueeze(1)
        elif sim_type == 'euclidean':
            # Euclidean distance similarity
            euclidean_distance = torch.sqrt(torch.sum((ref_embedding - student_embedding) ** 2, dim=1))
            similarity = (1 / (1 + euclidean_distance)).unsqueeze(1)
        
        return similarity
    
# ============================================================================= PIPELINE
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils.EarlyStopping import EarlyStopping
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import time
import logging

SEED = 42
torch.manual_seed(SEED)

# logging setup
logging.basicConfig(
    filename="training.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# init device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SBERTPipeline:
    def __init__(self, config, results, results_epoch):
        self.df = config['df']
        # tokenizer and model
        self.model = SiameseModel(config['model_name']).to(device)
        self.learning_rate = config['learning_rate']
        # optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        self.plateau_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        # step calculation for training data
        train_dataset, _, _ = self.split_dataset(0.8, 0.1, 0.1)
        num_training_steps = len(train_dataset) // config['batch_size'] * config['epochs']
        warmup_steps = int(config['warmup_ratio'] * num_training_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        # early stopping
        self.early_stopping = EarlyStopping(verbose=True, path='experiments/models/checkpoint.pt', patience=5)
        # loss function
        self.criterion = torch.nn.MSELoss()
        # other variable
        self.config = config
        self.results = results
        self.results_epoch = results_epoch

    def split_dataset(self, train_ratio, valid_ratio, test_ratio):
        print("run split dataset...")
        subset_dataset = self.df['dataset_num'].unique()
        splits = {}
        for subset in subset_dataset:
            # get data by dataset_num
            subset_df = self.df[self.df['dataset_num'] == subset]

            # split dataset
            train_df, temp_df = train_test_split(subset_df, test_size=(1 - train_ratio), random_state=SEED, shuffle=True)
            valid_df, test_df = train_test_split(temp_df, test_size=test_ratio / (valid_ratio + test_ratio), random_state=SEED, shuffle=True)

            # save split dataset
            splits[subset] = {
                'train': train_df,
                'valid': valid_df,
                'test': test_df,
            }
        
        train_dataset = pd.concat([splits[subset]['train'] for subset in subset_dataset])
        valid_dataset = pd.concat([splits[subset]['valid'] for subset in subset_dataset])
        test_dataset = pd.concat([splits[subset]['test'] for subset in subset_dataset])

        return train_dataset, valid_dataset, test_dataset
    
    def create_dataset(self, train_dataset, valid_dataset, test_dataset):
        print("create dataset run...")
        train_data = SBERTDataset(train_dataset)
        valid_data = SBERTDataset(valid_dataset)
        test_data = SBERTDataset(test_dataset)

        return train_data, valid_data, test_data
    
    @staticmethod
    def collate_fn(batch):
        reference_answers = [item['reference_answer'] for item in batch]
        student_answers = [item['student_answer'] for item in batch]
        scores = torch.tensor([item['score'] for item in batch], dtype=torch.float).unsqueeze(1)
        
        return {
            'reference_answers': reference_answers,
            'student_answers': student_answers,
            'scores': scores
        }
    
    def create_dataloader(self, train_data, valid_data, test_data):
        print("create dataloader run...")
        train_dataloader = DataLoader(train_data, batch_size=self.config['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(SEED), collate_fn=self.collate_fn)
        valid_dataloader = DataLoader(valid_data, batch_size=self.config['batch_size'], shuffle=False, generator=torch.Generator().manual_seed(SEED), collate_fn=self.collate_fn)
        test_dataloader = DataLoader(test_data, batch_size=self.config['batch_size'], shuffle=False, generator=torch.Generator().manual_seed(SEED), collate_fn=self.collate_fn)

        return train_dataloader, valid_dataloader, test_dataloader
    
    @staticmethod
    def save_model(model, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        logging.info(f"Model saved to {save_path}")

    def evaluate(self, dataloader, mode="validation"):
        if mode == 'testing':
            # self.model = SiameseScoringModel(self.config['model_name'], self.config['dropout']).to(device)
            self.model = SiameseModel(self.config['model_name']).to(device)
            checkpoint = torch.load('experiments/models/checkpoint.pt')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        self.model.eval()
        total_mse_loss = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch in dataloader:
                try:
                    # move to device
                    reference_answers = batch['reference_answers']
                    student_answers = batch['student_answers']
                    scores = batch['scores'].to(device)

                    # Forward pass
                    outputs = self.model(reference_answers, student_answers)
                    loss = self.criterion(outputs, scores)
                    
                    total_mse_loss += loss.item()
                    all_predictions.extend(outputs.detach().cpu().numpy())
                    all_targets.extend(batch['scores'].detach().cpu().numpy())
                except Exception as e:
                    logging.error(f"Error during {mode}: {str(e)}")
                    torch.cuda.empty_cache()

        avg_mse_loss = total_mse_loss / len(dataloader)
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        # change array dim from 1 to 0
        targets_flat = [t.item() for t in all_targets]
        predictions_flat = [p.item() for p in all_predictions]
        pearson_corr, _ = pearsonr(targets_flat, predictions_flat)

        return avg_mse_loss, mae, rmse, pearson_corr
    
    def training(self):
        # create dataset
        train_dataset, valid_dataset, test_dataset = self.split_dataset(0.8, 0.1, 0.1)
        train_data, valid_data, test_data = self.create_dataset(train_dataset, valid_dataset, test_dataset)
        train_dataloader, valid_dataloader, test_dataloader = self.create_dataloader(train_data, valid_data, test_data)

        # init start training time
        start_time = time.time()
        # experiment process
        epochs = self.config["epochs"]
        num_epochs = 0
        best_valid_metric = self.config["best_valid_pearson"] if self.config["best_valid_pearson"] is not None else float('-inf')
        best_model_path = os.path.join("experiments", "models", f"{self.config['model_name']}_best_model.pt")
        for epoch in range(epochs):
            num_epochs += 1
            print(f"====== Training Epoch {epoch + 1}/{epochs} ======")
            self.model.train()
            train_mse_loss = 0
            all_predictions = []
            all_targets = []
            for batch in train_dataloader:
                try:
                    self.optimizer.zero_grad()
                    # move to device
                    reference_answers = batch['reference_answers']
                    student_answers = batch['student_answers']
                    scores = batch['scores'].to(device) 

                    # get prediction
                    outputs = self.model(reference_answers, student_answers)
                    loss = self.criterion(outputs, scores)
                    
                    # backprop
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                    # save data for calculation
                    train_mse_loss += loss.item()
                    all_predictions.extend(outputs.detach().cpu().numpy())
                    all_targets.extend(batch['scores'].detach().cpu().numpy())
                except Exception as e:
                    logging.error(f"Error during training: {str(e)}")
                    torch.cuda.empty_cache()

            # calculate loss function and evaluation metrik
            avg_train_loss = train_mse_loss / len(train_dataloader)
            train_mae = mean_absolute_error(all_targets, all_predictions)
            train_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
            # change array dim from 1 to 0
            targets_flat = [t.item() for t in all_targets]
            predictions_flat = [p.item() for p in all_predictions]
            train_pearson, _ = pearsonr(targets_flat, predictions_flat)
            print(f"Epoch {epoch+1}/{epochs} - Avg training loss: {avg_train_loss:.4f}, MAE: {train_mae:.4}, RMSE: {train_rmse:.4}, Pearson Corr: {train_pearson:.4}")

            # =============== EVAL PROCESS
            valid_loss, valid_mae, valid_rmse, valid_pearson = self.evaluate(valid_dataloader, mode="validation")
            print(f"Avg validation loss: {valid_loss:.4f}, MAE: {valid_mae:.4}, RMSE: {valid_rmse:.4}, Pearson Corr: {valid_pearson:.4}")
            
            # update scheduler based on validation loss
            self.plateau_scheduler.step(valid_loss)

            # check early stopping
            self.early_stopping(val_loss=valid_loss, model=self.model)
            if(self.early_stopping.early_stop):
                logging.info(f"Early stopping triggered")
                print("Early stopping triggered")
                break
            
            # save model if get better pearson
            if valid_pearson > best_valid_metric:
                best_valid_metric = valid_pearson
                self.save_model(self.model, save_path=best_model_path)

            # save experiment result per epoch
            self.results_epoch.append({
                "config_id": self.config["config_id"],
                "epoch": epoch + 1,
                "train_mse": avg_train_loss,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "train_pearson": train_pearson,
                "valid_mse": valid_loss,
                "valid_mae": valid_mae,
                "valid_rmse": valid_rmse,
                "valid_pearson": valid_pearson,
                "learning_rate": self.learning_rate
            })

        # TESTING PROCESS
        test_loss, test_mae, test_rmse, test_pearson = self.evaluate(test_dataloader, mode="testing")
        print(f"Avg testing loss: {test_loss:.4f}, MAE: {test_mae:.4}, RMSE: {test_rmse:.4}, Pearson Corr: {test_pearson:.4}")

        # save experiment per configuration
        result = {
            "config_id": self.config.get("config_id"),
            "model_name": self.config.get("model_name"),
            "batch_size": self.config.get("batch_size"),
            "epochs": num_epochs,
            "learning_rate": self.config.get("learning_rate"),
            "warm_up": self.config['warmup_ratio'],
            "training_time": time.time() - start_time,
            "peak_memory": torch.cuda.max_memory_allocated(device) / (1024 ** 2),  # Convert to MB
            "test_mse": test_loss,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_pearson": test_pearson,
        }

        # Tambahkan hasil ke dalam list results
        self.results.append(result)
    
    @staticmethod
    def save_csv(data, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_exists = os.path.exists(filename)
        pd.DataFrame(data).to_csv(
            filename, mode="a" if file_exists else "w", header=not file_exists, index=False
        )

# ============================================================================= MAIN
import pandas as pd
import logging
import torch
import os

# df = pd.read_csv("data/aes_dataset_indo.csv") # tag nya nomor 3 di model
# df = pd.read_csv("data/aes_dataset_indo_siamese.csv") # tag nya nomor 1 di model
df = pd.read_csv("data/aes_dataset_new.csv") # tag nya nomor 4 di model
# df = pd.read_csv("data/aes_dataset_new_siamese.csv") # tag nya nomor 2 di model
# df = df[df['dataset'] == 'analisis_essay'][['reference_answer', 'answer', 'score', 'normalized_score', 'dataset', 'dataset_num']]
print(df.info())

# Check if the first file exists
df_result = None
if os.path.exists("experiments/results/new_dataset_mix_sbert.csv"):
    df_result = pd.read_csv("experiments/results/new_dataset_mix_sbert.csv")
    print(df_result['config_id'].iloc[-1])
else:
    print("File 'new_dataset_mix_sbert.csv' does not exist.")

idx = (df_result['config_id'].iloc[-1] + 1) if df_result is not None and not df_result.empty else 0  # index untuk setiap kombinasi
ROOT_DIR = os.getcwd()

results = []
results_epoch = []
df_result1 = None
# Check if the second file exists
if os.path.exists("experiments/results/new_dataset_mix_sbert_epoch.csv"):
    df_result1 = pd.read_csv("experiments/results/new_dataset_mix_sbert_epoch.csv")
    print(max(df_result1['valid_pearson']))
else:
    print("File 'new_dataset_mix_sbert_epoch.csv' does not exist.")

# set up hyperparamter
config = {
    "df": df,
    # "model_name": "indobenchmark/indobert-lite-base-p2",
    "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 100,
    "config_id": idx,
    "best_valid_pearson": max(df_result1['valid_pearson']) if df_result1 is not None and not df_result1.empty else float("-inf"),
    "warmup_ratio": 0.0
}

logging.info(
    f"Running configuration: config_id={idx}, model_name={config['model_name']}"
    f", batch_size={16}, epochs={100}, learning_rate={2e-5}"
)

print(
    f"\nRunning configuration: config_id={idx}, model_name={config['model_name']}"
    f", batch_size={16}, epochs={100}, learning_rate={2e-5}"
)

try:
    pipeline = SBERTPipeline(config, results, results_epoch)
    pipeline.training()

    # Save results
    # Dapatkan root project
    results_path = os.path.join(ROOT_DIR, "experiments/results/new_dataset_mix_sbert.csv")
    results_epoch_path = os.path.join(ROOT_DIR, "experiments/results/new_dataset_mix_sbert_epoch.csv")
    pipeline.save_csv(results, results_path)
    pipeline.save_csv(results_epoch, results_epoch_path)
except Exception as e:
    logging.error(f"Error in config_id={idx}: {str(e)}")
    print(f"Error in config_id={idx}: {str(e)}")
    torch.cuda.empty_cache()
finally:
    # Clear GPU memory after every configuration
    del pipeline.model
    del pipeline.optimizer
    torch.cuda.empty_cache()

idx += 1
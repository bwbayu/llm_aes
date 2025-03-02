import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.datasets.SBERTDataset import SBERTDataset
from src.models.SBERTRegressionModel import SBERTRegressionModel
from src.models.SiameseIndoBERTModel import SiameseIndoBERTModel
from src.pipelines.BERT_pipeline import BERTPipeline
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
        self.model = SiameseIndoBERTModel().to(device)
        # optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=config['learning_rate'])
        self.plateau_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
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
    
    def evaluate(self, dataloader, mode="validation"):
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
        best_valid_metric = self.config["best_valid_pearson"] if self.config["best_valid_pearson"] is not None else float('-inf')
        best_model_path = os.path.join("experiments", "models", f"{self.config['model_name']}_best_model.pt")
        for epoch in range(epochs):
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
            
            # save model if get better pearson
            if valid_pearson > best_valid_metric:
                best_valid_metric = valid_pearson
                BERTPipeline.save_model(self.model, save_path=best_model_path)

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
                "learning_rate": self.config['learning_rate']
            })

        # TESTING PROCESS
        test_loss, test_mae, test_rmse, test_pearson = self.evaluate(test_dataloader, mode="testing")
        print(f"Avg testing loss: {test_loss:.4f}, MAE: {test_mae:.4}, RMSE: {test_rmse:.4}, Pearson Corr: {test_pearson:.4}")

        # save experiment per configuration
        result = {
            "config_id": self.config.get("config_id"),
            "model_name": self.config.get("model_name"),
            "batch_size": self.config.get("batch_size"),
            "epochs": self.config.get("epochs"),
            "learning_rate": self.config.get("learning_rate"),
            "training_time": time.time() - start_time,
            "peak_memory": torch.cuda.max_memory_allocated(device) / (1024 ** 2),  # Convert to MB
            "test_mse": test_loss,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_pearson": test_pearson,
        }

        # Tambahkan hasil ke dalam list results
        self.results.append(result)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup
from src.datasets.BERTDataset import AutomaticScoringDataset
from src.models.BERTRegressionModel import RegressionModel
from src.utils.EarlyStopping import EarlyStopping
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

class BERTPipeline:
    def __init__(self, config, results, results_epoch):
        self.df = config['df']
        # tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        self.model = RegressionModel(config['model_name'], config['dropout'], freeze_transformer=True).to(device)
        # optimizer and scheduler
        if "learning_rate" in config and config['learning_rate'] is not None:
            self.learning_rate = config['learning_rate']
            self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        else:
            self.optimizer = AdamW([
                {'params': self.model.model.encoder.parameters(), 'lr': config.get('learning_rate_backbone', 2e-5)},  
                {'params': self.model.regression_layer.parameters(), 'lr': config.get('learning_rate_head', 1e-3)}
            ], weight_decay=0.01)
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
        self.early_stopping = EarlyStopping(patience=20, verbose=True, path='experiments/models/checkpoint.pt')
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
        train_data = AutomaticScoringDataset(train_dataset, self.tokenizer, use_reference=self.config['use_reference'])
        valid_data = AutomaticScoringDataset(valid_dataset, self.tokenizer, use_reference=self.config['use_reference'])
        test_data = AutomaticScoringDataset(test_dataset, self.tokenizer, use_reference=self.config['use_reference'])

        return train_data, valid_data, test_data
    
    def create_dataloader(self, train_data, valid_data, test_data):
        print("create dataloader run...")
        train_dataloader = DataLoader(train_data, batch_size=self.config['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(SEED))
        valid_dataloader = DataLoader(valid_data, batch_size=self.config['batch_size'], shuffle=False, generator=torch.Generator().manual_seed(SEED))
        test_dataloader = DataLoader(test_data, batch_size=self.config['batch_size'], shuffle=False, generator=torch.Generator().manual_seed(SEED))

        return train_dataloader, valid_dataloader, test_dataloader

    @staticmethod
    def save_model(model, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        logging.info(f"Model saved to {save_path}")

    def evaluate(self, dataloader, mode="validation"):
        if mode == 'testing':
            self.model = RegressionModel(self.config['model_name'], self.config['dropout'], freeze_transformer=True).to(device)
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
            for batchs in dataloader:
                try:
                    # move to device
                    batchs = {k: v.to(device) for k, v in batchs.items()}
                    predictions = self.model(
                        batchs['input_ids'], 
                        batchs['attention_mask'], 
                        batchs['token_type_ids']).squeeze(1)
                    loss = self.criterion(predictions, batchs['labels'])
                    if torch.isnan(loss):
                        print("⚠️ Warning: NaN detected in loss validation!")
                        print(f"Predictions: {predictions}")
                        print(f"Targets: {batchs['labels']}")
                        continue
                    total_mse_loss += loss.item()

                    all_predictions.extend(predictions.detach().cpu().numpy())
                    all_targets.extend(batchs['labels'].detach().cpu().numpy())
                except Exception as e:
                    logging.error(f"Error during {mode}: {str(e)}")
                    torch.cuda.empty_cache()

        avg_mse_loss = total_mse_loss / len(dataloader)
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        pearson, _ = pearsonr(all_targets, all_predictions)

        return avg_mse_loss, mae, rmse, pearson
    
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
            for batchs in train_dataloader:
                try:
                    self.optimizer.zero_grad()
                    # move to device
                    batchs = {k: v.to(device) for k, v in batchs.items()}

                    # get prediction
                    predictions = self.model(
                        batchs['input_ids'], 
                        batchs['attention_mask'], 
                        batchs['token_type_ids']).squeeze(1)
                    
                    # calculate loss function
                    loss = self.criterion(predictions, batchs['labels'])
                    if torch.isnan(loss):
                        print("⚠️ Warning: NaN detected in loss validation!")
                        print(f"Predictions: {predictions}")
                        print(f"Targets: {batchs['labels']}")
                        continue
                    
                    # backprop
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    # save data for calculation
                    train_mse_loss += loss.item()
                    all_predictions.extend(predictions.detach().cpu().numpy())
                    all_targets.extend(batchs['labels'].detach().cpu().numpy())
                except Exception as e:
                    logging.error(f"Error during training: {str(e)}")
                    torch.cuda.empty_cache()

            # calculate loss function and evaluation metrik
            avg_train_loss = train_mse_loss / len(train_dataloader)
            train_mae = mean_absolute_error(all_targets, all_predictions)
            train_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
            train_pearson, _ = pearsonr(all_targets, all_predictions)
            print(f"Epoch {epoch+1}/{epochs} - Avg training loss: {avg_train_loss:.4f}, MAE: {train_mae:.4}, RMSE: {train_rmse:.4}, Pearson Corr: {train_pearson:.4}")

            # EVALUATION PROCESS
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
                "learning_rate": self.learning_rate,
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
            # "learning_rate_backbone": self.config.get("learning_rate_backbone"),
            # "learning_rate_head": self.config.get("learning_rate_head"),
            "warm_up": self.config['warmup_ratio'],
            "dropout": self.config['dropout'],
            "training_time": time.time() - start_time,
            "peak_memory": torch.cuda.max_memory_allocated(device) / (1024 ** 2),  # Convert to MB
            "test_mse": test_loss,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_pearson": test_pearson,
            "use_reference": self.config['use_reference'],
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
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from longDataset import LongEssayDataset
from hierarchicalBert import HierarchicalBert
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import time
import logging
from tqdm import tqdm

# logging setup
logging.basicConfig(
    filename="training.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
# init device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainingBertPipeline:
    def __init__(self, config, results, results_epoch):
        self.df = config["df"]
        self.tokenizer = BertTokenizer.from_pretrained(config["model_name"])
        self.model = HierarchicalBert(config["model_name"]).to(device)
        # konfigurasi parameter
        self.optimizer = AdamW(self.model.parameters(), lr=config["learning_rate"])
        self.criterion = torch.nn.MSELoss()
        # 
        self.config = config
        self.results = results
        self.results_epoch = results_epoch

    def split_dataset(self, valid_size, test_size):
        subset_dataset = self.df['dataset'].unique()
        splits = {}
        # split dataset for each "category"
        for subset in subset_dataset:
            subset_df = self.df[self.df['dataset'] == subset]

            # split dataset
            train, temp = train_test_split(subset_df, test_size=valid_size, random_state=42)
            valid, test = train_test_split(temp, test_size=test_size, random_state=42)

            splits[subset] = {
                'train': train,
                'valid': valid,
                'test': test,
            }

        # concat dataset for each "category"
        train_dataset = pd.concat([splits[subset]['train'] for subset in subset_dataset])
        valid_dataset = pd.concat([splits[subset]['valid'] for subset in subset_dataset])
        test_dataset = pd.concat([splits[subset]['test'] for subset in subset_dataset])

        return train_dataset, valid_dataset, test_dataset

    def create_dataset(self, train_dataset, valid_dataset, test_dataset):
        train_data = LongEssayDataset(train_dataset, self.tokenizer, 512, self.config['overlapping'])
        valid_data = LongEssayDataset(valid_dataset, self.tokenizer, 512, self.config['overlapping'])
        test_data = LongEssayDataset(test_dataset, self.tokenizer, 512, self.config['overlapping'])

        return train_data, valid_data, test_data
    
    def create_dataloader(self, train_data, valid_data, test_data):
        train_dataloader = DataLoader(train_data, batch_size=self.config["batch_size"], collate_fn=lambda x: list(zip(*x)))
        valid_dataloader = DataLoader(valid_data, batch_size=self.config["batch_size"], collate_fn=lambda x: list(zip(*x)))
        test_dataloader = DataLoader(test_data, batch_size=self.config["batch_size"], collate_fn=lambda x: list(zip(*x)))

        return train_dataloader, valid_dataloader, test_dataloader
    
    @staticmethod
    def calculate_qwk(all_targets, all_predictions, min_score=0, max_score=100):
        rounded_predictions = [max(min(round(pred), max_score), min_score) for pred in all_predictions]
        rounded_targets = [max(min(round(target), max_score), min_score) for target in all_targets]
        return cohen_kappa_score(rounded_targets, rounded_predictions, weights="quadratic")
    
    def evaluate(self, dataloader, mode="validation"):
        self.model.eval()
        total_mse_loss = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch, targets in tqdm(dataloader, desc=f"Running {mode}", leave=False):
                try:
                    targets = torch.stack(targets).to(device)
                    predictions = self.model(batch).squeeze(1)
                    loss = self.criterion(predictions, targets)
                    total_mse_loss += loss.item()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                except Exception as e:
                    logging.error(f"Error during {mode}: {str(e)}")
                    torch.cuda.empty_cache()

        avg_mse_loss = total_mse_loss / len(dataloader)
        qwk_score = self.calculate_qwk(all_targets, all_predictions)
        return avg_mse_loss, qwk_score

    def run_training(self):
        # split dataset (70:20:10)
        train_dataset, valid_dataset, test_dataset = self.split_dataset(0.3, 0.3)
        
        # create dataset and dataLoader
        train_data, valid_data, test_data = self.create_dataset(train_dataset, valid_dataset, test_dataset)
        train_dataloader, valid_dataloader, test_dataloader = self.create_dataloader(train_data, valid_data, test_data)
        
        # init start training time
        start_time = time.time()
        # experiment process
        for epoch in range(self.config["epochs"]):
            print(f"====== Training Epoch {epoch + 1}/{self.config["epochs"]} ======")
            self.model.train()
            train_mse_loss = 0
            all_predictions = []
            all_targets = []
            for batch, targets in tqdm(train_dataloader, desc="Training", leave=False):
                try:
                    self.optimizer.zero_grad()
                    targets = torch.stack(targets).to(device)
                    predictions = self.model(batch).squeeze(1)
                    loss = self.criterion(predictions, targets)
                    loss.backward()
                    self.optimizer.step()
                    train_mse_loss += loss.item()
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                except Exception as e:
                    logging.error(f"Error during training: {str(e)}")
                    torch.cuda.empty_cache()

            avg_train_loss = train_mse_loss / len(train_dataloader)
            qwk_train = self.calculate_qwk(all_targets, all_predictions)
            print(f"Train Loss: {avg_train_loss:.4f}, Train QWK: {qwk_train:.4f}")

            valid_loss, valid_qwk = self.evaluate(valid_dataloader, mode="validation")
            print(f"Validation Loss: {valid_loss:.4f}, Validation QWK: {valid_qwk:.4f}")

            # save csv per training epoch
            self.results_epoch.append({
                "config_id": self.config["config_id"],
                "epoch": epoch + 1,
                "train_mse": avg_train_loss,
                "train_qwk": qwk_train,
                "valid_mse": valid_loss,
                "valid_qwk": valid_qwk
            })

        # run testing
        test_loss, test_qwk = self.evaluate(test_dataloader, mode="testing")
        print(f"Test Loss: {test_loss:.4f}, Test QWK: {test_qwk:.4f}")

        # save csv per training configuration
        self.results.append({
            "config_id": self.config["config_id"],
            "batch_size": self.config["batch_size"],
            "overlapping": self.config['overlapping'],
            "epochs": self.config["epochs"],
            "learning_rate": self.config["learning_rate"],
            "training_time": time.time() - start_time,
            "peak_memory": torch.cuda.max_memory_allocated(device) / (1024 ** 2), # Convert to MB
            "test_mse": test_loss,
            "test_qwk": test_qwk,
        })

    @staticmethod
    def save_csv(data, filename):
        file_exists = os.path.exists(filename)
        pd.DataFrame(data).to_csv(
            filename, mode="a" if file_exists else "w", header=not file_exists, index=False
        )

def main():
    # Load dataset
    df = pd.read_csv("dataset/aes_dataset.csv")

    # experiment result
    results = []
    results_epoch = []

    # hyperparameter configuration
    batch_sizes = [4, 8]
    overlappings = [0, 64, 128]
    epochs_list = [5, 10]
    learning_rates = [1e-5, 2e-5, 5e-5]
    idx = 0  # index untuk setiap kombinasi

    for batch_size in tqdm(batch_sizes, desc="Batch Size"):
        for overlapping in tqdm(overlappings, desc="Overlapping", leave=False):
            for num_epochs in tqdm(epochs_list, desc="Epochs", leave=False):
                for lr in tqdm(learning_rates, desc="Learning Rate", leave=False):
                    config = {
                        "df": df,
                        "model_name": "indobenchmark/indobert-lite-base-p2",
                        "overlapping": overlapping,
                        "batch_size": batch_size,
                        "learning_rate": lr,
                        "epochs": num_epochs,
                        "config_id": idx
                    }

                    logging.info(
                        f"Running configuration: config_id={idx}, batch_size={batch_size}, "
                        f"overlapping={overlapping}, epochs={num_epochs}, learning_rate={lr}"
                    )
                    
                    print(
                        f"\nRunning configuration: config_id={idx}, batch_size={batch_size}, "
                        f"overlapping={overlapping}, epochs={num_epochs}, learning_rate={lr}"
                    )
                    
                    try:
                        pipeline = TrainingBertPipeline(config, results, results_epoch)
                        pipeline.run_training()

                        # Save results
                        TrainingBertPipeline.save_csv(results, f"output/results_{idx}.csv")
                        TrainingBertPipeline.save_csv(results_epoch, f"output/results_epoch_{idx}.csv")
                    except Exception as e:
                        logging.error(f"Error in config_id={idx}: {str(e)}")
                        print(f"Error in config_id={idx}: {str(e)}")
                        torch.cuda.empty_cache()
                    finally:
                        # Clear GPU memory after every configuration
                        del pipeline.model
                        del pipeline.tokenizer
                        del pipeline.optimizer
                        torch.cuda.empty_cache()

                    idx += 1
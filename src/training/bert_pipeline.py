import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
# sys.path.append(os.path.join(os.getcwd(), '..', '..'))
import pandas as pd
import numpy as np
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from src.data.longDataset import LongEssayDataset
from src.models.hierarchicalBert import HierarchicalBert
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from peft import get_peft_model, LoraConfig
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

class TrainingBertPipeline:
    def __init__(self, config, results, results_epoch):
        self.df = config["df"]
        # implement fallback if there is model that require spesific Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        except TypeError: # "indobenchmark/indobert-lite-base-p2" --> this model only works with BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(config["model_name"])
        
        self.model = HierarchicalBert(config["model_name"]).to(device)
        # LoRA Configuration if there is a key that needed
        if config.get("lora_rank") is not None and config.get("lora_alpha") is not None:
            peft_config = LoraConfig(
                task_type="SEQ_CLS",
                inference_mode=False,
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=0.1,
                target_modules=["query", "key", "value"],
            )
            self.model = get_peft_model(self.model, peft_config)
        # konfigurasi parameter
        self.optimizer = AdamW(self.model.parameters(), lr=config["learning_rate"])
        self.criterion = torch.nn.MSELoss()
        # 
        self.config = config
        self.results = results
        self.results_epoch = results_epoch

    def split_dataset(self, valid_size, test_size):
        print("split dataset run...")
        subset_dataset = self.df['dataset'].unique()
        splits = {}
        # split dataset for each "category"
        for subset in subset_dataset:
            subset_df = self.df[self.df['dataset'] == subset]

            # split dataset
            train, temp = train_test_split(subset_df, test_size=valid_size, random_state=SEED)
            valid, test = train_test_split(temp, test_size=test_size, random_state=SEED)

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
        print("create dataset run...")
        train_data = LongEssayDataset(train_dataset, self.tokenizer, self.config['max_seq_len'], self.config['overlapping'], self.config['col_length'])
        valid_data = LongEssayDataset(valid_dataset, self.tokenizer, self.config['max_seq_len'], self.config['overlapping'], self.config['col_length'])
        test_data = LongEssayDataset(test_dataset, self.tokenizer, self.config['max_seq_len'], self.config['overlapping'], self.config['col_length'])

        return train_data, valid_data, test_data
    
    def create_dataloader(self, train_data, valid_data, test_data):
        print("create dataloader run...")
        train_dataloader = DataLoader(train_data, batch_size=self.config["batch_size"], collate_fn=lambda x: list(zip(*x)), shuffle=True, generator=torch.Generator().manual_seed(SEED),)
        valid_dataloader = DataLoader(valid_data, batch_size=self.config["batch_size"], collate_fn=lambda x: list(zip(*x)), shuffle=True, generator=torch.Generator().manual_seed(SEED),)
        test_dataloader = DataLoader(test_data, batch_size=self.config["batch_size"], collate_fn=lambda x: list(zip(*x)), shuffle=True, generator=torch.Generator().manual_seed(SEED),)

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
            for batch, targets in dataloader:
                try:
                    targets = torch.stack(targets).to(device)
                    predictions = self.model(batch).squeeze(1)
                    loss = self.criterion(predictions, targets)
                    total_mse_loss += loss.item()
                    all_predictions.extend(predictions.detach().cpu().numpy())
                    all_targets.extend(targets.detach().cpu().numpy())
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
        epochs = self.config["epochs"]
        for epoch in range(epochs):
            print(f"====== Training Epoch {epoch + 1}/{epochs} ======")
            self.model.train()
            train_mse_loss = 0
            all_predictions = []
            all_targets = []
            for batch, targets in train_dataloader:
                try:
                    self.optimizer.zero_grad()
                    targets = torch.stack(targets).to(device)
                    predictions = self.model(batch).squeeze(1)
                    loss = self.criterion(predictions, targets)
                    loss.backward()
                    self.optimizer.step()
                    train_mse_loss += loss.item()
                    all_predictions.extend(predictions.detach().cpu().numpy())
                    all_targets.extend(targets.detach().cpu().numpy())
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
        result = {
            "config_id": self.config.get("config_id"),
            "model_name": self.config.get("model_name"),
            "batch_size": self.config.get("batch_size"),
            "overlapping": self.config.get("overlapping"),
            "epochs": self.config.get("epochs"),
            "learning_rate": self.config.get("learning_rate"),
            "training_time": time.time() - start_time,
            "peak_memory": torch.cuda.max_memory_allocated(device) / (1024 ** 2),  # Convert to MB
            "test_mse": test_loss,
            "test_qwk": test_qwk,
        }

        # Jika ada konfigurasi LoRA, tambahkan ke hasil yang sama
        if self.config.get("lora_rank") is not None and self.config.get("lora_alpha") is not None:
            result.update({
                "lora_rank": self.config.get("lora_rank"),
                "lora_alpha": self.config.get("lora_alpha"),
            })

        # Tambahkan hasil ke dalam list results
        self.results.append(result)

    @staticmethod
    def save_csv(data, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_exists = os.path.exists(filename)
        pd.DataFrame(data).to_csv(
            filename, mode="a" if file_exists else "w", header=not file_exists, index=False
        )
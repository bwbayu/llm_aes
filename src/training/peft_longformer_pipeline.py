import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, LongformerConfig
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from src.data.dataset import EssayDataset
from src.models.regressionModelPeft import LongformerForRegression
from src.training.bert_pipeline import TrainingBertPipeline
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig
import time
import logging

SEED = 42
torch.manual_seed(SEED)

# logging setup
logging.basicConfig(
    filename="training_longformer_peft.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
# init device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LongFormerPipelinePeft:
    def __init__(self, config, results, results_epoch):
        self.df = config["df"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.model_config = LongformerConfig.from_pretrained(config["model_name"])
        self.model = LongformerForRegression.from_pretrained(config["model_name"], config=self.model_config).to(device)
        # LoRA Configuration if there is a key that needed
        if config.get("lora_rank") is not None and config.get("lora_alpha") is not None:
            peft_config = LoraConfig(
                task_type="SEQ_CLS",
                inference_mode=False,
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=0.1,
                target_modules=["query", "key", "value", "query_global", "key_global", "value_global",],
            )
            self.model = get_peft_model(self.model, peft_config)
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())

            print(f"Trainable Parameters: {trainable_params}")
            print(f"Total Parameters: {total_params}")
            print(f"Percentage Trainable: {(trainable_params / total_params) * 100:.2f}%")
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
        train_data = EssayDataset(train_dataset, self.tokenizer, self.config['max_seq_len'])
        valid_data = EssayDataset(valid_dataset, self.tokenizer, self.config['max_seq_len'])
        test_data = EssayDataset(test_dataset, self.tokenizer, self.config['max_seq_len'])

        return train_data, valid_data, test_data
    
    def create_dataloader(self, train_data, valid_data, test_data):
        print("create dataloader run...")
        train_dataloader = DataLoader(train_data, batch_size=self.config["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(SEED),)
        valid_dataloader = DataLoader(valid_data, batch_size=self.config["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(SEED),)
        test_dataloader = DataLoader(test_data, batch_size=self.config["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(SEED),)
        print("create dataloader done...")

        return train_dataloader, valid_dataloader, test_dataloader
    
    def save_model(self, save_path):
        """Save the model's state_dict to the specified path."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        logging.info(f"Model saved to {save_path}")
    
    def evaluate(self, dataloader, mode="validation"):
        self.model.eval()
        total_mse_loss = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batchs, targets in dataloader:
                try:
                    input_ids = batchs['input_ids'].to(device)
                    attention_mask = batchs['attention_mask'].to(device)
                    targets = targets.to(device)
                    predictions = self.model(input_ids, attention_mask).squeeze(1)
                    loss = self.criterion(predictions, targets)
                    total_mse_loss += loss.item()
                    all_predictions.extend(predictions.detach().cpu().numpy())
                    all_targets.extend(targets.detach().cpu().numpy())
                except Exception as e:
                    logging.error(f"Error during {mode}: {str(e)}")
                    torch.cuda.empty_cache()

        avg_mse_loss = total_mse_loss / len(dataloader)
        qwk_score = TrainingBertPipeline.calculate_qwk(all_targets, all_predictions)
        pearson_score = TrainingBertPipeline.calculate_pearson(all_targets, all_predictions)
        return avg_mse_loss, qwk_score, pearson_score
    
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
        best_valid_metric = self.config["best_valid_qwk"] if self.config["best_valid_qwk"] is not None else float('-inf')
        best_model_path = os.path.join("experiments", "models", f"{self.config['model_name']}_best_model_lora.pt")
        for epoch in range(epochs):
            print(f"====== Training Epoch {epoch + 1}/{epochs} ======")
            self.model.train()
            train_mse_loss = 0
            all_predictions = []
            all_targets = []
            for batchs, targets in train_dataloader:
                try:
                    self.optimizer.zero_grad()
                    input_ids = batchs['input_ids'].to(device)
                    attention_mask = batchs['attention_mask'].to(device)
                    targets = targets.to(device)
                    predictions = self.model(input_ids=input_ids, attention_mask=attention_mask).squeeze(1)
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
            qwk_train = TrainingBertPipeline.calculate_qwk(all_targets, all_predictions)
            pearson_train = TrainingBertPipeline.calculate_pearson(all_targets, all_predictions)
            print(f"Train Loss: {avg_train_loss:.4f}, Train QWK: {qwk_train:.4f}, Train Pearson: {pearson_train:.4f}")

            valid_loss, valid_qwk, valid_pearson = self.evaluate(valid_dataloader, mode="validation")
            print(f"Validation Loss: {valid_loss:.4f}, Validation QWK: {valid_qwk:.4f}, Validation Pearson: {valid_pearson:.4f}")

            # Save model if validation QWK improves
            if valid_qwk > best_valid_metric:
                best_valid_metric = valid_qwk
                self.save_model(save_path=best_model_path)

            # save csv per training epoch
            self.results_epoch.append({
                "config_id": self.config["config_id"],
                "epoch": epoch + 1,
                "train_mse": avg_train_loss,
                "train_qwk": qwk_train,
                "train_pearson": pearson_train,
                "valid_mse": valid_loss,
                "valid_qwk": valid_qwk,
                "valid_pearson": valid_pearson
            })

        # run testing
        test_loss, test_qwk, test_pearson = self.evaluate(test_dataloader, mode="testing")
        print(f"Test Loss: {test_loss:.4f}, Test QWK: {test_qwk:.4f}, Test Pearson: {test_pearson:.4f}")

        # save csv per training configuration
        result = {
            "config_id": self.config.get("config_id"),
            "model_name": self.config.get("model_name"),
            "batch_size": self.config.get("batch_size"),
            "epochs": self.config.get("epochs"),
            "learning_rate": self.config.get("learning_rate"),
            "max_seq_len": self.config.get("max_seq_len"),
            "training_time": time.time() - start_time,
            "peak_memory": torch.cuda.max_memory_allocated(device) / (1024 ** 2), # Convert to MB
            "test_mse": test_loss,
            "test_qwk": test_qwk,
            "test_pearson": test_pearson,
            "lora_rank": None,
            "lora_alpha": None
        }
    
        # Jika ada konfigurasi LoRA, tambahkan ke hasil yang sama
        if self.config.get("lora_rank") is not None and self.config.get("lora_alpha") is not None:
            result.update({
                "lora_rank": self.config.get("lora_rank"),
                "lora_alpha": self.config.get("lora_alpha"),
            })

        self.results.append(result)
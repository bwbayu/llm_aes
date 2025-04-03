# ============================================================================= DATASET
import torch
from torch.utils.data import Dataset
import re

class AutomaticScoringDataset(Dataset):
    def __init__(self, dataframe, tokenizer, use_reference=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.use_reference = use_reference  # Menentukan apakah menggunakan reference answer atau tidak

    def __len__(self):
        return len(self.dataframe)

    def preprocess_text(self, text):
        text = ' '.join(text.split())  # Hapus spasi berlebih
        text = text.lower()  # Ubah ke lowercase
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # Hapus karakter khusus
        return text

    def __getitem__(self, index):
        student_answer = self.preprocess_text(str(self.dataframe.iloc[index]['answer']))
        score = self.dataframe.iloc[index]['normalized_score']

        if self.use_reference and 'reference_answer' in self.dataframe.columns:
            reference_answer = self.preprocess_text(str(self.dataframe.iloc[index]['reference_answer']))
            encoding = self.tokenizer.encode_plus(
                reference_answer,
                student_answer,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            encoding = self.tokenizer.encode_plus(
                student_answer,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        encoding = {key: tensor.squeeze(0) for key, tensor in encoding.items()}
        encoding['labels'] = torch.tensor(score, dtype=torch.float)

        return encoding

    def get_max_length(self, index):
        """Menghitung panjang maksimum tokenized input berdasarkan apakah reference_answer digunakan atau tidak."""
        student_answer = str(self.dataframe.iloc[index]['answer'])

        if self.use_reference and 'reference_answer' in self.dataframe.columns:
            reference_answer = str(self.dataframe.iloc[index]['reference_answer'])
            encoding = self.tokenizer.encode_plus(
                reference_answer,
                student_answer,
                add_special_tokens=True,
                return_tensors='pt'
            )
        else:
            encoding = self.tokenizer.encode_plus(
                student_answer,
                add_special_tokens=True,
                return_tensors='pt'
            )

        return encoding['input_ids'].flatten().shape[0]
# ============================================================================= LAYER NORM
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x / norm)
    
def model_rms_norm(model, use_pretrained_weight=False):
    for name, module in model.named_children():
        if(isinstance(module, nn.LayerNorm)):
            # create rms norm
            hidden_size = module.normalized_shape[0]
            rms_norm = RMSNorm(hidden_size, eps=module.eps)
            # copy weight from pretrained
            if use_pretrained_weight:
                with torch.no_grad():
                    rms_norm.weight.copy_(module.weight)
            setattr(model, name, rms_norm)
        else:
            model_rms_norm(module, use_pretrained_weight)

    return model

class AdaNorm(nn.Module):
    def __init__(self, num_features, eps=0, C=1.0, k=1/10):
        super(AdaNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.C = C
        self.k = k

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(variance + self.eps) # add eps -> can be removed
        y = (x - mean) / std
        phi_y = self.C * (1 - self.k * y)
        phi_y = phi_y.detach()
        z = phi_y * y
        return z
    
def model_ada_norm(model, use_pretrained_weight=False):
    for name, module in model.named_children():
        if(isinstance(module, nn.LayerNorm)):
            # create rms norm
            # print("pretrained weight module", module.weight)
            hidden_size = module.normalized_shape[0]
            adanorm = AdaNorm(hidden_size, eps=0, C=1, k=1/10)
            setattr(model, name, adanorm)
        else:
            model_ada_norm(module, use_pretrained_weight)

    return model
    
class FilterResponseNormNd(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6, learnable_eps=False):
        super(FilterResponseNormNd, self).__init__()
        self.eps = nn.Parameter(torch.ones(1, 1, hidden_dim) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.ones(1, 1, hidden_dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.tau = nn.Parameter(torch.zeros(1, 1, hidden_dim)) 
    
    def forward(self, x):
        # nu2 = torch.mean(x**2, dim=[1, 2], keepdim=True)
        nu2 = torch.mean(x**2, dim=-1, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)
    
def model_fr_norm(model, use_pretrained_weight=False):
    for name, module in model.named_children():
        if(isinstance(module, nn.LayerNorm)):
            hidden_size = module.normalized_shape[0]
            fr_norm = FilterResponseNormNd(hidden_dim=hidden_size, eps=module.eps)
            if use_pretrained_weight:
                with torch.no_grad():
                    fr_norm.gamma.copy_(module.weight.view_as(fr_norm.gamma))
                    fr_norm.beta.copy_(module.bias.view_as(fr_norm.beta))
                    fr_norm.tau.fill_(0)
            setattr(model, name, fr_norm)
        else:
            model_fr_norm(module, use_pretrained_weight)

    return model

# ============================================================================= MODEL
import torch
import torch.nn as nn
from transformers import AutoModel, AlbertConfig, AlbertModel

class RegressionModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        # load config and model
        self.config = AlbertConfig.from_pretrained(model_name)
        self.model = AlbertModel(self.config)
        # load pretrained model
        pretrained_model = AutoModel.from_pretrained(model_name)
        # load pretrained weight
        self.model.load_state_dict(pretrained_model.state_dict()) 
        # add regression layer
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.regression_layer = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_embedding)
        score = self.regression_layer(x)
        return score
    
# ============================================================================= PIPELINE
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
        self.model = RegressionModel(config['model_name']).to(device)
        # load corresponding layer
        if config['layer_type'] == 'rms':
            self.model = model_rms_norm(self.model, use_pretrained_weight=config['use_pretrained_weight']).to(device)
        elif config['layer_type'] == 'ada':
            self.model = model_ada_norm(self.model, use_pretrained_weight=config['use_pretrained_weight']).to(device)
        elif config['layer_type'] == 'fr':
            self.model = model_fr_norm(self.model, use_pretrained_weight=config['use_pretrained_weight']).to(device)
        # print(self.model)
        # optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=config['learning_rate'])
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
        self.early_stopping = EarlyStopping(patience=10, verbose=True, path='experiments/models/checkpoint.pt')
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
            self.model = RegressionModel(self.config['model_name']).to(device)
            if config['layer_type'] == 'rms':
                self.model = model_rms_norm(self.model, use_pretrained_weight=config['use_pretrained_weight']).to(device)
            elif config['layer_type'] == 'ada':
                self.model = model_ada_norm(self.model, use_pretrained_weight=config['use_pretrained_weight']).to(device)
            elif config['layer_type'] == 'fr':
                self.model = model_fr_norm(self.model, use_pretrained_weight=config['use_pretrained_weight']).to(device)
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
        best_model_path = os.path.join("experiments", "models", "best_model", f"bert_regression_{self.config['layer_type']}_{self.config['use_pretrained_weight']}_{self.config['dataset_type']}.pt")
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
            "epochs": num_epochs,
            "learning_rate": self.config.get("learning_rate"),
            "warm_up": self.config['warmup_ratio'],
            "training_time": time.time() - start_time,
            "peak_memory": torch.cuda.max_memory_allocated(device) / (1024 ** 2),  # Convert to MB
            "test_mse": test_loss,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_pearson": test_pearson,
            "use_reference": self.config['use_reference'],
            "layer_norm": self.config['layer_type'],
            "use_pretrained_weight": self.config['use_pretrained_weight']
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

dataset_list = [("data/aes_dataset_indo.csv", "indo"), ("data/aes_dataset_new.csv", "new")]

for dataset_name, dataset_type in dataset_list:
    df = pd.read_csv(dataset_name)
    # df = pd.read_csv("data/aes_dataset_indo.csv") # tag nya nomor 1 di model
    # df = pd.read_csv("data/aes_dataset_new.csv") # tag nya nomor 2 di model
    # df = pd.read_csv("data/aes_dataset_5k_clean.csv")
    # df = df[df['dataset'] == 'analisis_essay'][['reference_answer', 'answer', 'score', 'normalized_score', 'dataset', 'dataset_num']]
    print(df.info())

    norm_layer_type = [('rms', True), ('rms', False), ('ada', False), ('fr', True), ('fr', False)]
    ROOT_DIR = os.getcwd()
    
    for layer, use_weight in norm_layer_type:
        if dataset_type == "indo" and layer == "rms" and use_weight == True:
            continue
        results = []
        results_epoch = []
        # Check if the first file exists
        df_result = None
        if os.path.exists(f"experiments/results/bert_{layer}_{use_weight}_{dataset_type}.csv"):
            df_result = pd.read_csv(f"experiments/results/bert_{layer}_{use_weight}_{dataset_type}.csv")
            print(df_result['config_id'].iloc[-1])
        else:
            print(f"File 'bert_{layer}_{use_weight}_{dataset_type}.csv' does not exist.")

        df_result1 = None
        # Check if the second file exists
        if os.path.exists(f"experiments/results/bert_{layer}_{use_weight}_{dataset_type}ch_new.csv"):
            df_result1 = pd.read_csv(f"experiments/results/bert_{layer}_{use_weight}_{dataset_type}ch_new.csv")
            print(max(df_result1['valid_pearson']))
        else:
            print(f"File 'bert_{layer}_{use_weight}_{dataset_type}ch_new.csv' does not exist.")

        # set up hyperparamter
        config = {
            "df": df,
            "model_name": "indobenchmark/indobert-lite-base-p2",
            "batch_size": 16,
            "learning_rate": 2e-5,
            "epochs": 100,
            "config_id": 0,
            "best_valid_pearson": max(df_result1['valid_pearson']) if df_result1 is not None and not df_result1.empty else float("-inf"),
            "warmup_ratio": 0.0,
            "use_reference": True,
            "layer_type": layer,
            "use_pretrained_weight": use_weight,
            "dataset_type": dataset_type,
        }

        logging.info(
            f"Running configuration: config_id={0}, model_name={config['model_name']}"
            f", batch_size={16}, epochs={100}, learning_rate={2e-5}"
        )

        print(
            f"\nRunning configuration: config_id={0}, model_name={config['model_name']}"
            f", batch_size={16}, epochs={100}, learning_rate={2e-5}"
        )

        try:
            pipeline = BERTPipeline(config, results, results_epoch)
            pipeline.training()

            # Save results
            # Dapatkan root project
            results_path = os.path.join(ROOT_DIR, f"experiments/results/bert_{layer}_{use_weight}_{dataset_type}.csv")
            results_epoch_path = os.path.join(ROOT_DIR, f"experiments/results/bert_{layer}_{use_weight}_{dataset_type}ch_new.csv")
            BERTPipeline.save_csv(results, results_path)
            BERTPipeline.save_csv(results_epoch, results_epoch_path)
        except Exception as e:
            logging.error(f"Error in config_id={0}: {str(e)}")
            print(f"Error in config_id={0}: {str(e)}")
            torch.cuda.empty_cache()
        finally:
            # Clear GPU memory after every configuration
            del pipeline.model
            del pipeline.tokenizer
            del pipeline.optimizer
            torch.cuda.empty_cache()
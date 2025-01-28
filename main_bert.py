import pandas as pd
from src.training.bert_pipeline import TrainingBertPipeline
import logging
import torch
import os

df = pd.read_csv("data/full_aes_dataset.csv")

results = []
results_epoch = []
batch_sizes = [4, 8]
overlappings = [0, 128, 256]
epochs_list = [5, 10]
learning_rates = [1e-5, 2e-5, 5e-5]
idx = 0  # index untuk setiap kombinasi

ROOT_DIR = os.getcwd()

for batch_size in batch_sizes:
    for overlapping in overlappings:
        for num_epochs in epochs_list:
            for lr in learning_rates:
                config = {
                    "df": df,
                    "model_name": "bert-base-uncased",
                    "overlapping": overlapping,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "epochs": num_epochs,
                    "config_id": idx,
                    "max_seq_len": 512,
                    "col_length": "bert_length",
                }

                logging.info(
                    f"Running configuration: config_id={idx}, model_name={config['model_name']}, batch_size={batch_size}, "
                    f"overlapping={overlapping}, epochs={num_epochs}, learning_rate={lr}"
                )
                
                print(
                    f"\nRunning configuration: config_id={idx}, model_name={config['model_name']}, batch_size={batch_size}, "
                    f"overlapping={overlapping}, epochs={num_epochs}, learning_rate={lr}"
                )
                
                try:
                    pipeline = TrainingBertPipeline(config, results, results_epoch)
                    pipeline.run_training()

                    # Save results
                    # Dapatkan root project
                    results_path = os.path.join(ROOT_DIR, "experiments/results/results.csv")
                    results_epoch_path = os.path.join(ROOT_DIR, "experiments/results/results_epoch.csv")
                    TrainingBertPipeline.save_csv(results, results_path)
                    TrainingBertPipeline.save_csv(results_epoch, results_epoch_path)
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
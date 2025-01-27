import pandas as pd
from src.training.longformer_pipeline import LongFormerPipeline
from src.training.bert_pipeline import TrainingBertPipeline
import logging
import torch
import os
import argparse

parser = argparse.ArgumentParser(description="Run LongFormer training with configurable hyperparameters.")

# Add arguments
parser.add_argument('--config_id', type=int, default=0, help='Configuration ID (default: 0)')
parser.add_argument('--max_seq_len', type=int, default=2048, help='Maximum sequence length (default: 2048)')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size (default: 2)')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate (default: 1e-5)')
parser.add_argument('--model_name', type=str, default='allenai/longformer-base-4096', help='Model name (default: allenai/longformer-base-4096)')
parser.add_argument('--results_dir', type=str, default='experiments/results', help='Directory to save results (default: experiments/results)')

# Parse arguments
args = parser.parse_args()

# Load dataset
df = pd.read_csv("data/full_aes_dataset.csv")
df.info()

# Initialize results lists
results = []
results_epoch = []

# Configuration dictionary
config = {
    "df": df,
    "model_name": args.model_name,
    "config_id": args.config_id,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "epochs": args.epochs,
    "max_seq_len": args.max_seq_len,
}

# Log and print configuration
logging.info(
    f"Running configuration: config_id={args.config_id}, model_name={args.model_name}, batch_size={args.batch_size}, "
    f"epochs={args.epochs}, learning_rate={args.learning_rate}"
)
print(
    f"\nRunning configuration: config_id={args.config_id}, model_name={args.model_name}, batch_size={args.batch_size}, "
    f"epochs={args.epochs}, learning_rate={args.learning_rate}"
)


try:
    pipeline = LongFormerPipeline(config, results, results_epoch)
    pipeline.run_training()

    results_path = os.path.join(args.results_dir, "results_longformer.csv")
    results_epoch_path = os.path.join(args.results_dir, "results_epoch_longformer.csv")
    TrainingBertPipeline.save_csv(results, results_path)
    TrainingBertPipeline.save_csv(results_epoch, results_epoch_path)
except Exception as e:
    logging.error(f"Error in config_id={args.config_id}: {str(e)}")
    print(f"Error in config_id={args.config_id}: {str(e)}")
    torch.cuda.empty_cache()
finally:
    # Clear GPU memory after training
    del pipeline.model
    del pipeline.tokenizer
    del pipeline.optimizer
    torch.cuda.empty_cache()

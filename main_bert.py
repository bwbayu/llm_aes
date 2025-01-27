import pandas as pd
from src.training.bert_pipeline import TrainingBertPipeline
import logging
import torch
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Run BERT training with configurable hyperparameters.")

# Add arguments
parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
parser.add_argument('--overlapping', type=int, default=128, help='Overlapping size (default: 128)')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs (default: 1)')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate (default: 1e-5)')
parser.add_argument('--config_id', type=int, default=0, help='Configuration ID (default: 0)')
parser.add_argument('--max_seq_len', type=int, default=512, help='Maximum sequence length (default: 512)')
parser.add_argument('--col_length', type=str, default='bert_length', help='Column name for length (default: bert_length)')
parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Model name (default: bert-base-uncased)')
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
    "overlapping": args.overlapping,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "epochs": args.epochs,
    "config_id": args.config_id,
    "max_seq_len": args.max_seq_len,
    "col_length": args.col_length,
}

# Log and print configuration
logging.info(
    f"Running configuration: config_id={args.config_id}, model_name={args.model_name}, batch_size={args.batch_size}, "
    f"overlapping={args.overlapping}, epochs={args.epochs}, learning_rate={args.learning_rate}"
)
print(
    f"\nRunning configuration: config_id={args.config_id}, model_name={args.model_name}, batch_size={args.batch_size}, "
    f"overlapping={args.overlapping}, epochs={args.epochs}, learning_rate={args.learning_rate}"
)

try:
    pipeline = TrainingBertPipeline(config, results, results_epoch)
    pipeline.run_training()

    results_path = os.path.join(args.results_dir, "results.csv")
    results_epoch_path = os.path.join(args.results_dir, "results_epoch.csv")
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

"""
How to run ? 
one configuration:
python3 train.py --batch_size 8 --overlapping 256 --epochs 3 --learning_rate 2e-5 --config_id 1

multiple configuration:
python3 train.py --batch_size 4 --overlapping 128 --epochs 1 --learning_rate 1e-5 --config_id 1 &
python3 train.py --batch_size 8 --overlapping 256 --epochs 3 --learning_rate 2e-5 --config_id 2 &
wait
"""
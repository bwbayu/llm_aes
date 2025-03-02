import torch
import logging
# logging setup
logging.basicConfig(
    filename="training.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, mode='min', verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss

        if self.best_score == None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # score/val loss didn't improve
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # score/val loss still improve
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
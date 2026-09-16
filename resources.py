import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIGURES_PATH = os.path.join(".", "figures")
DATA_PATH = os.path.join(".", "data")
DATASETS = [
    "Cora",
    "Chameleon"
]

INFO_STRINGS = {
    "CURRENT_EPOCH": "[INFO] Current Epoch Is {epoch}\n",
    "EPOCH_LOSS": "[INFO] Current Epoch Losses: Train - {train_loss}, Val - {val_loss}\n"
}
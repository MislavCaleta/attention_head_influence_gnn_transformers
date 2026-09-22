from src.data_loader import get_datasets
from src.models import LocalGraphTransformer, HybridGraphTransformer
from src.training import train_model
from src.settings import DEVICE, EXPERIMENT_CONFIG_PATH
from src.visualize import visualize_loss

import torch
from torch_geometric.data import Dataset
import yaml

with open(EXPERIMENT_CONFIG_PATH, "r") as f:
    experiment_config = yaml.safe_load(f)
    
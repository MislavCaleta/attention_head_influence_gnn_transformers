from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.data import Dataset
import os

from src.settings import DATA_PATH
from src.constants import DATASETS

def get_datasets() -> tuple[Dataset]:
    os.makedirs(DATA_PATH, exist_ok=True)
    cora_dataset = Planetoid(root=os.path.join(DATA_PATH, DATASETS[0]), name=DATASETS[0])
    chameleon_dataset = WikipediaNetwork(root=os.path.join(DATA_PATH, DATASETS[1]), name=DATASETS[1])

    return cora_dataset, chameleon_dataset
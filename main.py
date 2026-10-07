from src.data_loader import get_datasets
from src.models import LocalGraphTransformer, HybridGraphTransformer
from src.settings import DEVICE, EXPERIMENT_CONFIG_PATH
from src.experiment import run_all_experiments

import yaml

with open(EXPERIMENT_CONFIG_PATH, "r") as f:
    experiment_config = yaml.safe_load(f)
datasets = get_datasets()
model_definitions = [LocalGraphTransformer, HybridGraphTransformer]

run_all_experiments(
        datasets,
        model_definitions,
        experiment_config,
        True,
        "training_info.json"
)

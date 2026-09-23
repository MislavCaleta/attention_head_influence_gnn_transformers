from src.data_loader import get_datasets
from src.models import LocalGraphTransformer, HybridGraphTransformer
from src.settings import DEVICE, EXPERIMENT_CONFIG_PATH
from src.experiment import get_experiment_results

import yaml

with open(EXPERIMENT_CONFIG_PATH, "r") as f:
    experiment_config = yaml.safe_load(f)
datasets = get_datasets()
model_definitions = [LocalGraphTransformer, HybridGraphTransformer]

print(get_experiment_results(
    datasets,
    model_definitions,
    experiment_config
))

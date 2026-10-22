from src.data_loader import get_datasets
from src.models import LocalGraphTransformer, HybridGraphTransformer
from src.settings import DEVICE, EXPERIMENT_CONFIG_PATH
from src.experiment import run_all_experiments
from src.final_results import calculate_final_results
from src.visualize import create_final_figures
from src.significance import run_transformer_anova

import yaml

def main(
    train: bool = True,
    test: bool = True,
    final_results: bool = True,
    final_figures: bool = True,
    significance: bool = True
):

    with open(EXPERIMENT_CONFIG_PATH, "r") as f:
        experiment_config = yaml.safe_load(f)
    datasets = get_datasets()
    model_definitions = [LocalGraphTransformer, HybridGraphTransformer]

    if test:
        run_all_experiments(
            datasets,
            model_definitions,
            experiment_config,
            train,
            "training_info.json",
            "test_info.json"
        )
    
    if final_results:
        calculate_final_results(
            "test_info.json", 
            "final_results.json"
        )

    if final_figures:
        create_final_figures(
            "final_results.json",
            experiment_config
        )

    if significance:
        run_transformer_anova(
            "final_results.json"
        )

if __name__ == "__main__":
    main(
        train = False,
        test = False,
        final_results = False,
        final_figures = True,
        significance = True
    )

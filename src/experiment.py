from src.training import train_model
from src.settings import (
    EXPERIMENT_RESULTS_PATH,
    ALL_MODELS_PATH
)
from src.strings import EXCEPTIONS 

import os
import torch
import json
from torch_geometric.data import Dataset
from pathlib import Path

def get_training_results(
    datasets: tuple[Dataset],
    model_definitions: list[torch.nn.Module],
    experiment_config: dict,
    current_seed: int
) -> dict[str, dict[int, dict]]:
    experiment_information = dict()
    for dataset in datasets:
        experiment_information[dataset.name] = dict()
        for attention_heads in experiment_config["model"]["num_attention_heads"][:2]:
            models = [
                model_definition(	
                    layers = experiment_config["model"]["num_layers"],
                    attention_heads = attention_heads,
                    input_channels = dataset[0].x.size()[1],
                    hidden_channels = experiment_config["model"]["hidden_dim"],
                    classes = dataset.num_classes	
                )
                for model_definition in model_definitions
            ]
            
            current_training_information = [
                train_model(
                    model = model,
                    dataset = dataset,
                    epochs = experiment_config["training"]["epochs"],
                    device = "cuda",
                    optimizer = torch.optim.Adam(
                        lr = experiment_config["training"]["learning_rate"],
                        params = model.parameters()
                    ),
                    criterion = torch.nn.CrossEntropyLoss()			
                ) 
                for model in models			
            ]
            
            os.makedirs(ALL_MODELS_PATH, exist_ok = True)
            for info in current_training_information:
                model_info = {
                    "model_class": type(info[0]).__name__,
                    "state_dict": info[0].state_dict()
                }
                torch.save(
                    model_info,
                    os.path.join(
                        ALL_MODELS_PATH, 
                        f"{type(info[0]).__name__}_{dataset.name}_{attention_heads}-{current_seed}.pt"
                    )
                )
            
            experiment_information[dataset.name][attention_heads] = {
                type(current_training_information[0][0]).__name__: {
                    "train_losses": current_training_information[0][1],
                    "val_losses": current_training_information[0][2]
                },
                type(current_training_information[1][0]).__name__: {
                    "train_losses": current_training_information[1][1],
                    "val_losses": current_training_information[1][2]
                }
            }
            
    return experiment_information

def test_models(
        model_definitions: [torch.nn.Module]
):  
    test_results = dict()
    if Path(ALL_MODELS_PATH).is_dir(): 
       model_list = list(os.listdir(ALL_MODELS_PATH))

       for name_seed in model_list:
           model_info = torch.load(ALL_MODELS_PATH / name_seed)
           model_definition = [model_definition for model_definition in model_definitions if model_definition.__name__ == model_info["model_class"]][0]
           model = model_definition(
            #pass corresponding arguments
           )
           model.load_state_dict(model_info["state_dict"])

           name, seed = name_seed.split("-")
           if name not in test_results.keys:
                test_results[name] = dict()
           test_results[name][seed] = get_metrics(model) #implement get metrics function
    else:
        raise FileNotFoundError(
            EXCEPTIONS["NO_MODELS_DIRECTORY"]
        )

    return test_results

def run_all_experiments(
    datasets: tuple[Dataset],
    model_definitions: list[torch.nn.Module],
    experiment_config: dict,
    retrain: bool,
    results_training_file: str,
    results_test_file: str
):  
    if retrain:
        training_results_by_seed = dict()
        for seed in experiment_config["seeds"][:2]:
            training_results = get_training_results(
                datasets,
                model_definitions,
                experiment_config,
                seed
            )
            training_results_by_seed[seed] = training_results
    
        os.makedirs(EXPERIMENT_RESULTS_PATH, exist_ok = True)
        with open(EXPERIMENT_RESULTS_PATH / results_training_file, "w") as f:
            json.dump(training_results_by_seed, f, indent=4)

    try:
        test_results = test_models(model_definitions)
    except FileNotFoundError as e:
        print(f"{type(e).__name__}: {e}")

    os.makedirs(EXPERIMENT_RESULTS_PATH, exist_ok = True)
    with open(EXPERIMENT_RESULTS_PATH / results_test_file, "w") as f:
        json.dump(test_results, f, indent=4)

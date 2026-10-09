from src.settings import EXPERIMENT_RESULTS_PATH 

import json
import torch

def calculate_final_results(
    test_results_file: str,
    final_results_file: str
):
    with open(EXPERIMENT_RESULTS_PATH / test_results_file, "r") as f:
        test_results = json.load(f)
    
    avg_per_model = dict()
    for model in test_results.keys():
        seeds = list(test_results[model].keys())
        avg_per_model[model] = torch.unsqueeze(torch.tensor(test_results[model][seeds[0]]), dim = 0)
        for seed in seeds[1:]:
            avg_per_model[model] = torch.concat(
                    [
                        avg_per_model[model],
                        torch.unsqueeze(torch.tensor(test_results[model][seed]), dim = 0)
                    ],
                    dim = 0
            )
        avg_per_model[model] = torch.concat(
                [
                    torch.mean(avg_per_model[model], dim = 0, keepdim = True),
                    torch.std(avg_per_model[model], dim = 0, keepdim = True)
                ],
                dim = 0
        ).tolist()
    
    with open(EXPERIMENT_RESULTS_PATH / final_results_file, "w") as f:
        json.dump(avg_per_model, f, indent = 4)


            

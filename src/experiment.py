from src.training import train_model
import torch
from torch_geometric.data import Dataset

def get_experiment_results(
    datasets: tuple[Dataset],
    model_definitions: list[torch.nn.Module],
    experiment_config: dict
) -> dict[str, dict[int, dict]]:
    experiment_information = dict()
    for dataset in datasets:
        experiment_information[dataset.name] = dict()
        for attention_heads in experiment_config["model"]["num_attention_heads"][:2]:
            models = [
                model_definition(	
                    layers = experiment_config["model"]["num_layers"],
                    attention_heads = attention_heads,
                    input_channels = datasets[0].x.size()[1],
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


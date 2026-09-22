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

datasets = get_datasets()
model_definitions = [LocalGraphTransformer, HybridGraphTransformer]

for dataset in datasets:
	for attention_heads in experiment_config["model"]["num_attention_heads"]:
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
		
		training_information = [
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

print(training_information)		

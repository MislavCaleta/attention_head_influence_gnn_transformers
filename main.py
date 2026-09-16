from data_loader import get_datasets
from models import LocalGraphTransformer
from training import train_model
from resources import DEVICE

import torch

cora, chameleon = get_datasets()
model = LocalGraphTransformer(
    layers=3,
    attention_heads=3,
    input_channels=cora[0].x.size()[1],
    hidden_channels=16,
    classes = cora.num_classes
)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
train_model(
    model = model,
    dataset = cora,
    epochs = 50,
    device = "cuda",
    criterion = torch.nn.CrossEntropyLoss(),
    optimizer = optimizer
)

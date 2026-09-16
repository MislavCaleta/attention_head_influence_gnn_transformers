import torch
from torch_geometric.data import Dataset
from typing import Literal

from resources import INFO_STRINGS

def train_model(
    model:  torch.nn.Module,
    dataset: Dataset,
    epochs: int,
    device: Literal["cpu", "cuda"],
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer
) -> tuple[torch.nn.Module, list[float]]:
    train_losses = []
    val_losses = []
    
    training_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")  
    model = model.to(training_device)
    data = dataset[0].to(training_device)

    for epoch in range(epochs):
        print(INFO_STRINGS["CURRENT_EPOCH"].format(epoch=epoch))
        model.train()
        outputs = model(data.x, data.edge_index)
        loss = criterion(outputs[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(outputs[data.val_mask], data.y[data.val_mask])
            val_losses.append(val_loss.item())

        print(INFO_STRINGS["EPOCH_LOSS"].format(train_loss=loss.item(), val_loss=val_loss.item()))

    return model, train_losses, val_losses



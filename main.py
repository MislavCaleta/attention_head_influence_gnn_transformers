from src.data_loader import get_datasets
from src.models import LocalGraphTransformer
from src.training import train_model
from src.settings import DEVICE
from src.visualize import visualize_loss

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
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)
model, train_losses, val_losses = train_model(
    model = model,
    dataset = cora,
    epochs = 50,
    device = "cuda",
    criterion = torch.nn.CrossEntropyLoss(),
    optimizer = optimizer
)
visualize_loss(
    {
        "train loss": train_losses,
        "val loss": val_losses
    },
    output_figure_name="train_val_loss.jpeg"
)

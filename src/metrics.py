import torch
import torchmetrics.functional as tmf
from torch_geometric.data import Dataset

def get_metrics(
    model: torch.nn.Module,
    dataset: Dataset
) -> list[float]:
    metrics = []
    if dataset[0].test_mask.ndim > 1:
        test_mask = dataset[0].test_mask[:, 0]
    else:
        test_mask = dataset[0].test_mask

    model_outputs = model(dataset[0].x, dataset[0].edge_index);
    predictions = torch.argmax(model_outputs, dim=1)
    
    acc = tmf.accuracy(
        predictions[test_mask],
        dataset[0].y[test_mask],
        task = "multiclass",
        num_classes = dataset.num_classes
    ).item()
    metrics.append(acc)

    macro_f1 = tmf.f1_score(
        predictions[test_mask],
        dataset[0].y[test_mask],
        task = "multiclass",
        num_classes = dataset.num_classes,
        average = "macro",
    ).item()
    metrics.append(macro_f1)

    return metrics

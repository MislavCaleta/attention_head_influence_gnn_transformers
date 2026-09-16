import matplotlib.pyplot as plt
import os
from torch_geometric.utils import homophily
from torch_geometric.data import Dataset

from src.settings import WORKING_FIGURES_PATH, PLOT_STYLE_PATH

plt.style.use(PLOT_STYLE_PATH)

def visualize_loss(
        loss_info: dict[str, list[float]],
        output_figure_name: str
):
    fig, ax = plt.subplots()
    for key, value in loss_info.items():
        ax.plot(value, label=key)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss Value")
    ax.legend()
    os.makedirs(WORKING_FIGURES_PATH, exist_ok=True)
    fig.savefig(os.path.join(WORKING_FIGURES_PATH, output_figure_name))

def collect_homophily(
        datasets: list[Dataset]
) -> list[float]:
    homophily_measures = []

    for dataset in datasets:
        homophily_measures.append(
            homophily(
                dataset[0].edge_index,
                dataset[0].y,
                method="edge"
            )
        )

    return homophily_measures

def visualize_homophily(
        datasets: list[Dataset],
        output_figure_name: str
):
    homophilies = collect_homophily(datasets)
    dataset_names = [dataset.name for dataset in datasets]
    colors = []
    for score in homophilies:
        if score < 0.5:
            colors.append("red")
        else:
            colors.append("blue")

    fig, ax = plt.subplots()
    bars = ax.bar(x=dataset_names, height=homophilies, color=colors)
    ax.bar_label(bars)
    os.makedirs(WORKING_FIGURES_PATH, exist_ok=True)
    fig.savefig(fname=os.path.join(WORKING_FIGURES_PATH, output_figure_name))
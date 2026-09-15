from data_loader import get_datasets
from torch_geometric.utils import homophily
from torch_geometric.data import Dataset
from matplotlib import pyplot as plt
import os

from resources import FIGURES_PATH

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
    fig.savefig(fname=os.path.join(FIGURES_PATH, output_figure_name))
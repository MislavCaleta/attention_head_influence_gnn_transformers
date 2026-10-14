import matplotlib.pyplot as plt
import os
from torch_geometric.utils import homophily
from torch_geometric.data import Dataset
import json
import numpy as np

from src.settings import WORKING_FIGURES_PATH, PLOT_STYLE_PATH, EXPERIMENT_RESULTS_PATH, EXPERIMENT_FIGURES_PATH 

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

def create_final_figure(
    dataset_results: dict,
    figure_name: str,
    metric_index: int,
    metric_name: str,
    bar_width: float = 0.35
):  
    fig, ax = plt.subplots()
    for i, (model_name, results) in enumerate(dataset_results.items()):
        results_by_head = []
        std_by_head = []
        heads = []
        for result in results:
            results_by_head.append(result[0][0][metric_index])
            std_by_head.append(result[0][1][metric_index])
            heads.append(result[1])
        offset = (i - 0.5) * bar_width
        ax.bar(np.array(heads) + offset, results_by_head, bar_width, label=model_name)
    
    ax.set_xlabel("Number of Attention Heads")
    ax.set_ylabel(f"Score: {metric_name}")
    ax.set_xticks(heads)
    ax.legend()
    
    os.makedirs(EXPERIMENT_FIGURES_PATH, exist_ok=True)
    fig.savefig(fname=EXPERIMENT_FIGURES_PATH / figure_name)

def create_final_figures(
    final_results_file: str,
    experiment_config: dict
):
    with open(EXPERIMENT_RESULTS_PATH / final_results_file) as f:
        final_results = json.load(f)
    
    per_dataset = {}
    for key, value in final_results.items():
        model, dataset_name, heads = key.split("_")
        if dataset_name not in per_dataset.keys():
            per_dataset[dataset_name] = dict()
        if model not in per_dataset[dataset_name].keys():
            per_dataset[dataset_name][model] = []
        per_dataset[dataset_name][model].append([value, int(heads)])

    for model_dict in per_dataset.values():
        for model_name, results_list in model_dict.items():
            # this sorts the results by heads
            model_dict[model_name].sort(key=lambda x: x[1])

    metrics_dict = experiment_config["metrics"]
    for metric_name, metric_index in metrics_dict.items():
        for dataset_name, model_results in per_dataset.items():
            create_final_figure(
                model_results,
                f"results_{dataset_name}_{metric_name}.png",
                metric_index,
                metric_name
            )

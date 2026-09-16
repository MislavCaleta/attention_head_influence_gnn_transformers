from pathlib import Path
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = Path(__file__).resolve().parent.parent
WORKING_FIGURES_PATH = ROOT_DIR / "outputs" / "figures"
DATA_PATH = ROOT_DIR / "data"
PLOT_STYLE_PATH = ROOT_DIR / "resources" / "plot_style.mplstyle"
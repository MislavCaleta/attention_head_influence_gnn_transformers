import json
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from src.settings import EXPERIMENT_RESULTS_PATH

def run_transformer_anova(filename="final_results.json"):
    """
    Loads JSON experiment results, parses model configurations, datasets, and head counts,
    and performs a Multi-way ANOVA analysis with interactions (Heads x Architecture and Heads x Dataset).
    """
    # 1. Load data from the JSON file
    try:
        with open(EXPERIMENT_RESULTS_PATH / filename, "r") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return

    parsed_rows = []

    # 2. Parse JSON keys into a structured tabular format
    for key, value in raw_data.items():
        parts = key.rsplit("_", 2)
        architecture = parts[0] 
        dataset = parts[1]      
        head_number = int(parts[2]) 
        
        acc_mean = value[0][0]
        f1_mean = value[0][1]
        
        parsed_rows.append({
            "Architecture": architecture,
            "Dataset": dataset,
            "Heads": head_number,
            "Acc_Mean": acc_mean,
            "F1_Mean": f1_mean
        })

    df = pd.DataFrame(parsed_rows)

    # 3. Multi-way ANOVA with interactions for F1-Score
    print("="*60)
    print("STATISTICAL ANALYSIS: ANOVA FOR F1-SCORE (WITH INTERACTIONS)")
    print("="*60)
    model_f1 = ols('F1_Mean ~ C(Heads) + C(Architecture) + C(Dataset) + C(Heads):C(Architecture) + C(Heads):C(Dataset)', data=df).fit()
    anova_f1 = sm.stats.anova_lm(model_f1, typ=2)
    print(anova_f1)

    # 4. Same analysis for Accuracy
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS: ANOVA FOR ACCURACY (WITH INTERACTIONS)")
    print("="*60)
    model_acc = ols('Acc_Mean ~ C(Heads) + C(Architecture) + C(Dataset) + C(Heads):C(Architecture) + C(Heads):C(Dataset)', data=df).fit()
    anova_acc = sm.stats.anova_lm(model_acc, typ=2)
    print(anova_acc)

    # 5. Summary of significance 
    print("\n" + "="*60)
    print("INTERPRETATION OF KEY FACTORS AND INTERACTIONS:")
    print("="*60)
    for name, anova_res in [("F1-Score", anova_f1), ("Accuracy", anova_acc)]:
        print(f"\n[{name}]")
        for index, row in anova_res.iterrows():
            p_val = row['PR(>F)']
            if pd.isna(p_val):
                continue
            significance = "STATISTICALLY SIGNIFICANT (p < 0.05)" if p_val < 0.05 else "Not statistically significant (p >= 0.05)"
            print(f" - {index}: p-value = {p_val:.4f} --> {significance}")

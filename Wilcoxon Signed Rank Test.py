# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 09:19:51 2026
Test for Wilcoxon signed-rank test
@author: djy41
"""
import pandas as pd
from scipy.stats import wilcoxon

c3u_path = "results/MULTI_COIL_20_C3U-MVC_runs_20260606_092357.csv"
baseline_path = "results/MULTI_COIL_20_SVDMVC_runs.csv"

baseline = pd.read_csv(baseline_path, sep=";")
c3u = pd.read_csv(c3u_path)

# Keep only the columns needed for the test
c3u = c3u[["seed", "stage2_acc", "stage2_nmi", "stage2_pur"]].rename(
    columns={
        "stage2_acc": "c3u_acc",
        "stage2_nmi": "c3u_nmi",
        "stage2_pur": "c3u_pur",
    }
)

baseline = baseline[["seed", "stage2_acc", "stage2_nmi", "stage2_pur"]].rename(
    columns={
        "stage2_acc": "baseline_acc",
        "stage2_nmi": "baseline_nmi",
        "stage2_pur": "baseline_pur",
    }
)

# Match rows by seed
paired = pd.merge(c3u, baseline, on="seed", how="inner")
paired = paired.sort_values("seed")

print(paired)

# Paired Wilcoxon signed-rank test
for metric in ["acc", "nmi", "pur"]:
    c3u_vals = paired[f"c3u_{metric}"].values
    base_vals = paired[f"baseline_{metric}"].values

    stat_two, p_two = wilcoxon(c3u_vals, base_vals, alternative="two-sided")
    stat_greater, p_greater = wilcoxon(c3u_vals, base_vals, alternative="greater")

    print(f"\nMetric: {metric.upper()}")
    print(f"  C3U mean:      {c3u_vals.mean():.4f}")
    print(f"  Baseline mean: {base_vals.mean():.4f}")
    print(f"  Wilcoxon two-sided: W={stat_two:.4f}, p={p_two:.6f}")
    print(f"  Wilcoxon greater:   W={stat_greater:.4f}, p={p_greater:.6f}")

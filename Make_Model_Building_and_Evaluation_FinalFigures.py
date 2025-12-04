'''
Create final presentation figures from the Model Building and Evaluation task for the 
3 sea surface salinity (SSS) regions: low, medium and high SSS variability.
Also compute statistical evidence to support results and analyses
'''
# Clear variables and console
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic('reset', '-f')
get_ipython().run_line_magic('clear', '')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import seaborn as sns
from scipy.stats import pearsonr, probplot, shapiro
from scipy import stats
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

# Paths
IN_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(IN_DIR, "model_sss_region_outputs")
DATA_DIR_4x3 = os.path.join(IN_DIR, "model_sss_region_outputs_4x3Folds")
LOW_DIR = os.path.join(DATA_DIR, "low")
MED_DIR = os.path.join(DATA_DIR, "med")
HIGH_DIR = os.path.join(DATA_DIR, "high")
OUT_DIR = os.path.join(IN_DIR, "model_figs_output")
os.makedirs(OUT_DIR, exist_ok=True)
#------------------------------------------------------------------------------------------
# Plot region comparisons
# Load All Input Files 
cross_region_winners_df = pd.read_csv(os.path.join(DATA_DIR, "cross_region_winners.csv"))
cross_region_r2_df = pd.read_csv(os.path.join(DATA_DIR, "cross_region_r2_comparison.csv"))
cross_region_bline_comp_df = pd.read_csv(os.path.join(DATA_DIR, "cross_region_baseline_comparison.csv"))
cross_region_bline_comp_4x3Fold_df = pd.read_csv(os.path.join(DATA_DIR_4x3, "cross_region_baseline_comparison.csv"))

nested_cv_all_models_LowVar_df = pd.read_csv(glob.glob(os.path.join(LOW_DIR, "nested_cv_all_models.csv"))[0])
nested_cv_all_models_MedVar_df = pd.read_csv(glob.glob(os.path.join(MED_DIR, "nested_cv_all_models.csv"))[0])
nested_cv_all_models_HighVar_df = pd.read_csv(glob.glob(os.path.join(HIGH_DIR, "nested_cv_all_models.csv"))[0])

diag_data_foldFullData_LowVar_df = pd.read_csv(glob.glob(os.path.join(LOW_DIR, "diag_data_foldFullData_*_FULL.csv"))[0])
diag_data_foldFullData_MedVar_df = pd.read_csv(glob.glob(os.path.join(MED_DIR, "diag_data_foldFullData_*_FULL.csv"))[0])
diag_data_foldFullData_HighVar_df = pd.read_csv(glob.glob(os.path.join(HIGH_DIR, "diag_data_foldFullData_*_FULL.csv"))[0])

final_perm_importance_LowVar_df = pd.read_csv(glob.glob(os.path.join(LOW_DIR, "final_perm_importance_*.csv"))[0])
final_perm_importance_MedVar_df = pd.read_csv(glob.glob(os.path.join(MED_DIR, "final_perm_importance_*.csv"))[0])
final_perm_importance_HighVar_df = pd.read_csv(glob.glob(os.path.join(HIGH_DIR, "final_perm_importance_*.csv"))[0])

# Get Partial dependence data
# Define all region configs
pdp_data = {}

region_dirs = {
    "LowVar": LOW_DIR,
    "MedVar": MED_DIR,
    "HighVar": HIGH_DIR,
}

for region, region_dir in region_dirs.items():
    # infer model name from the permutation-importance file
    perm_file = glob.glob(os.path.join(region_dir, "final_perm_importance_*.csv"))[0]
    model_name = os.path.splitext(os.path.basename(perm_file))[0].split("_")[-1]

    # grab ALL PDP files for that winning model in this region
    pattern = os.path.join(region_dir, f"pdp_data_*_{model_name}.csv")
    pdp_files = sorted(glob.glob(pattern))

    print(f"{region}: found {len(pdp_files)} PDP files for model {model_name}")
    pdp_data[region] = [pd.read_csv(f) for f in pdp_files]


# Define consistent custom color palette for all models
model_palette = {
    "SVR": "#D62728",          # red
    "RandomForest": "#1F77B4", # blue
    "XGBoost": "#9467BD",   # purple
    "ElasticNet": "#2CA02C",  # green
    "Baseline_Linear": "#8C564B"  # brown
}

# Define consistent hue order for all models
hue_order = ["ElasticNet", "RandomForest", "SVR", "XGBoost", "Baseline_Linear"]
#------------------------------------------------------------------------------------------
# Plot cross region R2 comparisons
plt.figure(figsize=(10, 5))
sns.barplot(data=cross_region_r2_df, x="Region", y="Mean_R2", hue="model",
    hue_order=hue_order, palette=model_palette, edgecolor="black")

plt.title("Cross-Region R² Comparison Across All Models", fontsize=18, fontweight='bold')
plt.ylabel("Mean R²", fontsize=16)
plt.xlabel("Region", fontsize=16)

plt.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=11, ncol=2, frameon=True)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# Add stars for best model per region
for region, group in cross_region_r2_df.groupby('Region'):
    best_model_idx = group['Mean_R2'].idxmax()
    best_model = group.loc[best_model_idx, 'model']
    best_r2 = group.loc[best_model_idx, 'Mean_R2']

    # x-position of the region
    x_pos = list(cross_region_r2_df['Region'].unique()).index(region)

    # Use fixed hue_order 
    n_hues = len(hue_order)
    hue_index = hue_order.index(best_model)
    bar_width = 0.8 / n_hues
    x_offset = -0.4 + (hue_index + 0.5) * bar_width

    # Adjust vertical position for positive/negative bars
    if best_r2 >= 0:
        y_pos = best_r2 + 0.01
        va = 'bottom'
    else:
        y_pos = best_r2 - 0.01
        va = 'top'

    plt.text(x_pos + x_offset, y_pos, '★', color='black', fontsize=14, ha='center', va=va)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cross_region_r2_comparison.png"), dpi=300)
plt.show()
#------------------------------------------------------------------------------------------

# Plot cross region RMSE comparisons
plt.figure(figsize=(10, 5))
sns.barplot(data=cross_region_r2_df, x="Region", y="Mean_RMSE", hue="model", 
            hue_order=hue_order, palette=model_palette, edgecolor="black")

plt.title("Cross-Region RMSE Comparison Across All Models", fontsize=18, fontweight='bold')
plt.ylabel("Mean RMSE", fontsize=16)
plt.xlabel("Region", fontsize=16)

plt.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=12, ncol=2, frameon=True)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# Add stars for best model per region
for region, group in cross_region_r2_df.groupby('Region'):
    best_model_idx = group['Mean_RMSE'].idxmin()
    best_model = group.loc[best_model_idx, 'model']
    best_r2 = group.loc[best_model_idx, 'Mean_RMSE']

    x_pos = list(cross_region_r2_df['Region'].unique()).index(region)
    n_hues = len(hue_order)
    hue_index = hue_order.index(best_model)
    bar_width = 0.8 / n_hues
    x_offset = -0.4 + (hue_index + 0.5) * bar_width

    # Adjust vertical position based on sign of RMSE
    if best_r2 >= 0:
        y_pos = best_r2 + 0.01
        va = 'bottom'
    else:
        y_pos = best_r2 - 0.01
        va = 'top'

    plt.text(x_pos + x_offset, y_pos, '★', color='black', fontsize=14, ha='center', va=va)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cross_region_RMSE_comparison.png"), dpi=300)
plt.show()
#------------------------------------------------------------------------------------------

# Plot cross region MAE comparisons
plt.figure(figsize=(10, 5))
sns.barplot(data=cross_region_r2_df, x="Region", y="Mean_MAE", hue="model", 
            hue_order=hue_order, palette=model_palette, edgecolor="black")

plt.title("Cross-Region MAE Comparison Across All Models", fontsize=18, fontweight='bold')
plt.ylabel("Mean MAE", fontsize=16)
plt.xlabel("Region", fontsize=16)

plt.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=12, ncol=2, frameon=True)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# Add stars for best model per region
for region, group in cross_region_r2_df.groupby('Region'):
    best_model_idx = group['Mean_MAE'].idxmin()
    best_model = group.loc[best_model_idx, 'model']
    best_r2 = group.loc[best_model_idx, 'Mean_MAE']

    x_pos = list(cross_region_r2_df['Region'].unique()).index(region)
    n_hues = len(hue_order)
    hue_index = hue_order.index(best_model)
    n_hues = len(hue_order)
    bar_width = 0.8 / n_hues
    x_offset = -0.4 + (hue_index + 0.5) * bar_width

    # Adjust vertical position based on sign of MAE
    if best_r2 >= 0:
        y_pos = best_r2 + 0.01
        va = 'bottom'
    else:
        y_pos = best_r2 - 0.01
        va = 'top'

    plt.text(x_pos + x_offset, y_pos, '★', color='black', fontsize=14, ha='center', va=va)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cross_region_MAE_comparison.png"), dpi=300)
plt.show()
#------------------------------------------------------------------------------------------

# Plot cross region model winners
region_order = ["low", "med", "high"]

plt.figure(figsize=(8, 5))
sns.barplot(data=cross_region_winners_df, x="Region", y="Mean_R2", hue="Winner", order=region_order, 
    dodge=False, hue_order=hue_order, palette=model_palette, edgecolor="black")

plt.title("Best Model per Region", fontsize=18, fontweight='bold')
plt.xlabel("Region", fontsize=16)
plt.ylabel("Mean R²", fontsize=16)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# Keep only labels that appear in the current dataframe
handles, labels = plt.gca().get_legend_handles_labels()
valid_labels = cross_region_winners_df['Winner'].unique()  
filtered = [(h, l) for h, l in zip(handles, labels) if l in valid_labels]

# Recreate legend with only valid entries
plt.legend(*zip(*filtered), loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=11, frameon=True)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cross_region_winners.png"), dpi=300)
plt.show()
#------------------------------------------------------------------------------------------

# Plot cross region model MAE reduction
region_order = ["low", "med", "high"]

plt.figure(figsize=(8, 5))
sns.barplot(data=cross_region_bline_comp_df, x="Region", y="MAE_Reduction_%", hue="Best_Model",
    order=region_order, dodge=False, hue_order=hue_order, palette=model_palette, edgecolor="black")

plt.title("Cross-Region % MAE Reduction (Best vs Baseline)", fontsize=18, fontweight='bold')
plt.ylabel("% Reduction in MAE", fontsize=16)
plt.xlabel("Region", fontsize=16)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# Keep only labels that appear in the current dataframe
handles, labels = plt.gca().get_legend_handles_labels()
valid_labels = cross_region_bline_comp_df['Best_Model'].unique()  
filtered = [(h, l) for h, l in zip(handles, labels) if l in valid_labels]

# Recreate legend with only valid entries
plt.legend(*zip(*filtered), loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=11, frameon=True)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cross_region_mae_reduction.png"), dpi=300)
plt.show()

#------------------------------------------------------------------------------------------
#Statistical support for the model output analyses: compute Mean ± SD and pairwise statistical tests per Region
def summarize_and_compare_region(region_name, file_path, baseline_model="Baseline_Linear"):
    df = pd.read_csv(file_path)

    # Clean numeric columns
    for col in ["R2", "RMSE", "MAE", "best_inner_R2"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace(["nan", "None", ""], np.nan)
            .astype(float)
        )

    print(f"\n{region_name}: {len(df)} rows, {df['model'].nunique()} models, {df['fold'].nunique()} folds")

    # Summary table of mean ± SD 
    summary_rows = []
    for model, group in df.groupby("model"):
        summary_rows.append({
            "Region": region_name,
            "Model": model,
            "R² ± SD": f"{group['R2'].mean():.2f} ± {group['R2'].std(ddof=1):.2f}",
            "RMSE ± SD": f"{group['RMSE'].mean():.3f} ± {group['RMSE'].std(ddof=1):.3f}",
            "MAE ± SD": f"{group['MAE'].mean():.3f} ± {group['MAE'].std(ddof=1):.3f}",
            "mean_R2": group["R2"].mean(),
            "mean_MAE": group["MAE"].mean()
        })
    summary_df = pd.DataFrame(summary_rows)

    # Baseline and paired stats
    results = []
    baseline = df[df["model"] == baseline_model].sort_values("fold")

    for model in df["model"].unique():
        if model == baseline_model:
            continue

        comp = df[df["model"] == model].sort_values("fold")
        merged = pd.merge(baseline, comp, on="fold", suffixes=("_base", "_comp"))

        if len(merged) > 1:
            merged["ΔR2"] = merged["R2_comp"] - merged["R2_base"]
            merged["ΔMAE_pct"] = 100 * (merged["MAE_base"] - merged["MAE_comp"]) / merged["MAE_base"]

            # Paired tests
            t_stat_r2, t_p_r2 = stats.ttest_rel(merged["R2_comp"], merged["R2_base"])
            w_stat_r2, w_p_r2 = stats.wilcoxon(merged["R2_comp"], merged["R2_base"])

            t_stat_mae, t_p_mae = stats.ttest_rel(merged["MAE_base"], merged["MAE_comp"])
            w_stat_mae, w_p_mae = stats.wilcoxon(merged["MAE_base"], merged["MAE_comp"])

            # One-sample t-test on %MAE reduction vs 0 
            if len(merged["ΔMAE_pct"].dropna()) > 1:
                t_stat_pct, p_val_pct = stats.ttest_1samp(merged["ΔMAE_pct"].dropna(), 0)
            else:
                p_val_pct = np.nan

            # Effect sizes 
            d_r2 = merged["ΔR2"].mean() / merged["ΔR2"].std(ddof=1) if merged["ΔR2"].std(ddof=1) != 0 else np.nan
            d_mae = merged["ΔMAE_pct"].mean() / merged["ΔMAE_pct"].std(ddof=1) if merged["ΔMAE_pct"].std(ddof=1) != 0 else np.nan

            mean_diff_r2, std_diff_r2 = merged["ΔR2"].mean(), merged["ΔR2"].std(ddof=1)
            mean_diff_mae, std_diff_mae = merged["ΔMAE_pct"].mean(), merged["ΔMAE_pct"].std(ddof=1)

            # Combined significance
            if not np.isnan(p_val_pct):
                if p_val_pct < 0.05:
                    sig_combined = "Statistically Significant (p<0.05) *"
                elif p_val_pct < 0.1:
                    sig_combined = "Marginally Significant (p<0.1) †"
                elif d_mae > 0.8:
                    sig_combined = "Practically Significant (large d)"
                else:
                    sig_combined = "Not Significant"
            else:
                sig_combined = ""

        else:
            t_p_r2 = w_p_r2 = d_r2 = np.nan
            t_p_mae = w_p_mae = d_mae = np.nan
            mean_diff_r2 = std_diff_r2 = mean_diff_mae = std_diff_mae = np.nan
            p_val_pct = np.nan
            sig_combined = ""

        results.append({
            "Region": region_name,
            "Model": model,
            "ΔR² ± SD": f"{mean_diff_r2:.3f} ± {std_diff_r2:.3f}" if not np.isnan(mean_diff_r2) else "",
            "p (R² t-test)": f"{t_p_r2:.3f}" if not np.isnan(t_p_r2) else "",
            "p (R² Wilcoxon)": f"{w_p_r2:.3f}" if not np.isnan(w_p_r2) else "",
            "Effect size (R² d)": f"{d_r2:.2f}" if not np.isnan(d_r2) else "",
            "ΔMAE (% mean ± SD)": f"{mean_diff_mae:.2f}% ± {std_diff_mae:.2f}%" if not np.isnan(mean_diff_mae) else "",
            "p (MAE t-test)": f"{t_p_mae:.3f}" if not np.isnan(t_p_mae) else "",
            "p (MAE Wilcoxon)": f"{w_p_mae:.3f}" if not np.isnan(w_p_mae) else "",
            "p (ΔMAE%)": f"{p_val_pct:.3f}" if not np.isnan(p_val_pct) else "",
            "Effect size (MAE d)": f"{d_mae:.2f}" if not np.isnan(d_mae) else "",
            "Significance (Combined)": sig_combined,
            "t (R²)": f"{t_stat_r2:.2f}" if not np.isnan(t_stat_r2) else "",
            "df (R²)": len(merged) - 1 if len(merged) > 1 else "",
            "t (ΔMAE%)": f"{t_stat_pct:.2f}" if 't_stat_pct' in locals() and not np.isnan(t_stat_pct) else "",
            "df (ΔMAE%)": len(merged["ΔMAE_pct"].dropna()) - 1 if len(merged["ΔMAE_pct"].dropna()) > 1 else ""
        })

    stats_df = pd.DataFrame(results)

    # Significance markers
    def sig_marker(p_str):
        try:
            p = float(p_str)
            if p < 0.01: return "***"
            elif p < 0.05: return "**"
            elif p < 0.1: return "*"
            else: return ""
        except:
            return ""
    stats_df["Sig (R²)"] = stats_df["p (R² t-test)"].apply(sig_marker)
    stats_df["Sig (MAE)"] = stats_df["p (MAE t-test)"].apply(sig_marker)

    # Merge summaries
    merged_df = pd.merge(summary_df, stats_df, on=["Region", "Model"], how="left")

    # Identify best model (by mean R2)
    best_model = summary_df.loc[summary_df["mean_R2"].idxmax(), "Model"]
    best_entry = stats_df[stats_df["Model"] == best_model].copy()
    best_entry["Model"] = f"**BestModel_{best_model}**"

    # Compute aggregate percent MAE reduction
    mean_base = summary_df.loc[summary_df["Model"] == baseline_model, "mean_MAE"].values[0]
    mean_best = summary_df.loc[summary_df["Model"] == best_model, "mean_MAE"].values[0]
    agg_pct_reduction = 100 * (mean_base - mean_best) / mean_base
    best_entry["Aggregate % MAE Reduction (Best vs Baseline)"] = f"{agg_pct_reduction:.2f}%"

    # Append best summary row
    merged_df = pd.concat([merged_df, best_entry], ignore_index=True)
    merged_df.drop(columns=["mean_R2", "mean_MAE"], inplace=True, errors="ignore")

    return merged_df

# Run for each region
low_df  = summarize_and_compare_region("Low",  os.path.join(DATA_DIR, "low",  "nested_cv_all_models.csv"))
med_df  = summarize_and_compare_region("Med",  os.path.join(DATA_DIR, "med",  "nested_cv_all_models.csv"))
high_df = summarize_and_compare_region("High", os.path.join(DATA_DIR, "high", "nested_cv_all_models.csv"))

# Combine all the regions 
all_summary = pd.concat([low_df, med_df, high_df], ignore_index=True)

# Save final output
out_path = os.path.join(OUT_DIR, "all_regions_model_comparison_summary_with_pctMAE_bestAgg.csv")
all_summary.to_csv(out_path, index=False)


#------------------------------------------------------------------------------------------
# Compare cross-region % MAE reductionplot 4x3 fold and 6x4 fold
# Extract relevant columns
df_4x3 = cross_region_bline_comp_4x3Fold_df[['Region', 'MAE_Reduction_%']]
df_6x4 = cross_region_bline_comp_df[['Region', 'MAE_Reduction_%']]

# Rename for merging clarity
df_4x3.rename(columns={'MAE_Reduction_%': 'MAE_4x3'}, inplace=True)
df_6x4.rename(columns={'MAE_Reduction_%': 'MAE_6x4'}, inplace=True)

# Merge on Region
merged = pd.merge(df_4x3, df_6x4, on='Region', how='inner')

# Sort regions in logical order
region_order = ['low', 'med', 'high']
merged['Region'] = pd.Categorical(merged['Region'], categories=region_order, ordered=True)
merged.sort_values('Region', inplace=True)

# Plot 
bar_width = 0.35
x = np.arange(len(merged))

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - bar_width/2, merged['MAE_4x3'], bar_width, label='4×3', color='#E69F00')
bars2 = ax.bar(x + bar_width/2, merged['MAE_6x4'], bar_width, label='6×4', color='#56B4E9')

# Annotate bars with percentages
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height - 0.3, f"{height:.0f}%",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_title("Cross-Region % MAE Reduction: Best vs Baseline", fontsize=19, fontweight='bold')
ax.set_xlabel("Region", fontsize=16, fontweight='bold')
ax.set_ylabel("% Reduction in MAE", fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(merged['Region'], fontsize=16)
ax.legend(title="FOLD", frameon=True, fontsize=16, prop={'weight': 'bold'}, title_fontproperties={'weight': 'bold'})

ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "Compare_FOLDS.png")
plt.savefig(out_path, dpi=300)
plt.show()

#------------------------------------------------------------------------------------------
#Compute Statistical evidence for choosing 6x4 folds over 4x3 folds
REGIONS = ["low", "med", "high"]
BASELINE = "Baseline_Linear"

# Helper Functions
def clean_numeric(df, cols=("R2", "RMSE", "MAE", "best_inner_R2")):
    for c in cols:
        df[c] = (
            df[c].astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace(["nan", "None", ""], np.nan)
            .astype(float)
        )
    return df

def compute_mae_reductions(base_dir):
    """Compute per-fold %MAE reductions for all models vs Baseline_Linear."""
    all_reductions = []
    for region in REGIONS:
        path = os.path.join(base_dir, region, "nested_cv_all_models.csv")
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        df = pd.read_csv(path)
        df = clean_numeric(df)
        base = df[df["model"] == BASELINE][["fold", "MAE"]].rename(columns={"MAE": "MAE_base"})

        for model in df["model"].unique():
            if model == BASELINE:
                continue

            comp = df[df["model"] == model][["fold", "MAE"]].rename(columns={"MAE": "MAE_model"})
            merged = pd.merge(base, comp, on="fold", how="inner").dropna()

            merged["pct_reduction"] = 100 * (merged["MAE_base"] - merged["MAE_model"]) / merged["MAE_base"]
            all_reductions.extend(merged["pct_reduction"].tolist())

    return np.array(all_reductions, dtype=float)

def hedges_g(x, y):
    """Unpaired effect size (Hedges’ g)."""
    nx, ny = len(x), len(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_p = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    d = (np.mean(y) - np.mean(x)) / s_p if s_p > 0 else np.nan
    J = 1 - (3 / (4 * (nx + ny) - 9))
    return J * d

def bootstrap_ci(a, b, B=10000, seed=42):
    """Bootstrap 95% CI for mean difference (b - a)."""
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(b, len(b), True).mean() - rng.choice(a, len(a), True).mean() for _ in range(B)]
    return np.percentile(diffs, [2.5, 97.5])

# Compute all MAE reductions
mae_4x3 = compute_mae_reductions(DATA_DIR_4x3)
mae_6x4 = compute_mae_reductions(DATA_DIR)

# Global comparison 
t_stat, t_p = stats.ttest_ind(mae_6x4, mae_4x3, equal_var=False)
u_stat, u_p = stats.mannwhitneyu(mae_6x4, mae_4x3, alternative="two-sided")
g = hedges_g(mae_4x3, mae_6x4)
ci_lo, ci_hi = bootstrap_ci(mae_4x3, mae_6x4)

fold_summary = pd.DataFrame([{
    "4×3 Mean %MAE Reduction": np.mean(mae_4x3),
    "4×3 SD": np.std(mae_4x3, ddof=1),
    "6×4 Mean %MAE Reduction": np.mean(mae_6x4),
    "6×4 SD": np.std(mae_6x4, ddof=1),
    "Mean Difference (6×4 − 4×3)": np.mean(mae_6x4) - np.mean(mae_4x3),
    "95% CI (Bootstrap)": f"[{ci_lo:.2f}, {ci_hi:.2f}]",
    "Welch t-stat": t_stat,
    "Welch t-test p": t_p,
    "Mann–Whitney U stat": u_stat,
    "Mann–Whitney U p": u_p,
    "Effect size (Hedges’ g)": g,
    "N (4×3)": len(mae_4x3),
    "N (6×4)": len(mae_6x4)
}])

# Save the results 
out_csv = os.path.join(OUT_DIR, "global_fold_scheme_comparison_withSD.csv")
fold_summary.to_csv(out_csv, index=False)

# Visualization 
fig, ax = plt.subplots(figsize=(6, 6))
data = [mae_4x3, mae_6x4]
labels = ['4×3 Folds', '6×4 Folds']

# Boxplot
box = ax.boxplot(data, labels=labels, patch_artist=True, medianprops=dict(color="black"))
colors = ["#E69F00", "#56B4E9"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

# Compute mean ± SD
means = [np.mean(mae_4x3), np.mean(mae_6x4)]
stds = [np.std(mae_4x3, ddof=1), np.std(mae_6x4, ddof=1)]

# Place labels below the lowest data point (including outliers)
mins = [np.min(mae_4x3), np.min(mae_6x4)]

for i, (m, s, min_val) in enumerate(zip(means, stds, mins), start=1):
    ax.text(i, min_val - 5, f"Mean = {m:.2f} ± {s:.2f}%", 
            ha='center', va='top', fontsize=13, color="black", weight='bold')

# Add p-values + effect size at top
y_top = max(max(mae_4x3), max(mae_6x4)) * 0.85
ax.text(1.5, y_top,
        f"Welch p = {t_p:.3f}\nMann–Whitney p = {u_p:.3f}\nHedges’ g = {g:.2f}",
        ha='center', fontsize=13, color='black')

ax.set_ylabel("MAE Reduction (%)", fontsize=16)
ax.set_title("Comparison of CV Fold Configurations", fontsize=16, weight='bold')
ax.grid(True, linestyle="--", alpha=0.4)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
y_min = min(np.min(mae_4x3), np.min(mae_6x4))
y_max = max(np.max(mae_4x3), np.max(mae_6x4))
ax.set_ylim(y_min - 0.1 * abs(y_max - y_min), y_max + 0.05 * abs(y_max - y_min))

fig_path = os.path.join(OUT_DIR, "global_fold_scheme_comparison_boxplot.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()

#------------------------------------------------------------------------------------------
# Compare actual and predicted SSS, and Q–Q plot of residuals
region_data = {
    "Low Var Region": diag_data_foldFullData_LowVar_df,
    "Med Var Region": diag_data_foldFullData_MedVar_df,
    "High Var Region": diag_data_foldFullData_HighVar_df
}

for region_name, df in region_data.items():
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    resid = df["residual"]

    # Compute correlation
    r_value, _ = pearsonr(y_true, y_pred)
    r_label = f"r = {r_value:.2f}"

    # Create 3 subplots 
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # (1) Predicted vs Actual
    sns.scatterplot(ax=axes[0], x=y_true, y=y_pred, alpha=0.7, s=50, color="royalblue", edgecolor="black")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--")
    axes[0].set_title(f"{region_name}: Predicted vs Actual", fontsize=16, fontweight="bold")
    axes[0].text(0.05, 0.9, r_label, transform=axes[0].transAxes,
                 fontsize=16, color="black", fontweight="bold",
                 bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    axes[0].set_xlabel("Actual SSS (PSU)", fontsize=16)
    axes[0].set_ylabel("Predicted SSS (PSU)", fontsize=16)

    # (2) Residuals vs Predicted
    sns.scatterplot(ax=axes[1], x=y_pred, y=resid, alpha=0.7, s=50, color="royalblue", edgecolor="black")
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_title("Residuals vs Predicted", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("Predicted SSS (PSU)", fontsize=16)
    axes[1].set_ylabel("Residuals (PSU)", fontsize=16)

    # (3) Q–Q Plot of Residuals
    probplot(resid, dist="norm", plot=axes[2])
    axes[2].get_lines()[0].set_alpha(0.6)   
    axes[2].get_lines()[1].set_color("red") 
    axes[2].set_title("Q–Q Plot of Residuals", fontsize=16, fontweight="bold")

    axes[2].set_xlabel("Theoretical Quantiles", fontsize=16)
    axes[2].set_ylabel("Sample Quantiles", fontsize=16)
    axes[2].tick_params(axis='both', which='major', labelsize=16)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    fname = f"diagnostic_{region_name.replace(' ', '').replace(':', '')}.png"
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=300, bbox_inches="tight")
    plt.show()
    
#------------------------------------------------------------------------------------------
# Statistical diagnostics for error analysis 
# Compute statistics for each region
summary_rows = []

for region_name, df in region_data.items():
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    resid = df["residual"]

    # Correlation and R2
    r_value, _ = pearsonr(y_true, y_pred)
    r2_value = r_value ** 2

    # Residual summary
    mean_resid = np.mean(resid)
    std_resid = np.std(resid, ddof=1)

    # Shapiro–Wilk test for normality
    shapiro_p = shapiro(resid)[1] if len(resid) >= 3 else np.nan

    # Breusch–Pagan test for homoscedasticity
    X = sm.add_constant(y_pred)
    bp_p = het_breuschpagan(resid, X)[1] if len(resid) >= 3 else np.nan

    # Store results
    summary_rows.append({
        "Region": region_name,
        "r": r_value,
        "R²": r2_value,
        "Mean Residual (PSU)": mean_resid,
        "SD Residual (PSU)": std_resid,
        "Shapiro–Wilk p": shapiro_p,
        "Breusch–Pagan p": bp_p
    })

# Combine into a table
resid_stats_df = pd.DataFrame(summary_rows)

# Round for display
resid_stats_df = resid_stats_df.round({
    "r": 2, "R²": 2,
    "Mean Residual (PSU)": 3, "SD Residual (PSU)": 3,
    "Shapiro–Wilk p": 3, "Breusch–Pagan p": 3
})

# Save to CSV 
out_path = os.path.join(OUT_DIR, "residual_diagnostics_summary.csv")
resid_stats_df.to_csv(out_path, index=False)

#------------------------------------------------------------------------------------------
# Plot feature importance
# Dictionary of regions and dataframes 
region_dfs = {"Low Var Region": final_perm_importance_LowVar_df,
    "Med Var Region": final_perm_importance_MedVar_df,
    "High Var Region": final_perm_importance_HighVar_df}

# Cleaning function 
def clean_feature_name(feature):
    feature = feature.replace("low_", "").replace("med_", "").replace("high_", "")
    feature = feature.replace("sla", "SSH").replace("sst", "SST").replace("EPR", "E-P+R")
    return feature

# Loop over regions 
for region_name, df in region_dfs.items():
    df = df.copy()
    df["feature"] = df["feature"].apply(clean_feature_name)

    # Plot
    plt.figure(figsize=(8, 10))
    sns.barplot(data=df.sort_values("importance", ascending=False).head(10),
        x="importance", y="feature", orient="h", palette="viridis")

    plt.title(f"{region_name}: Feature Importance (Top 10)", fontsize=21, fontweight='bold', x=0.33)
    plt.xlabel("Mean Importance (Δ score)", fontsize=18, fontweight='bold')
    plt.ylabel("Feature", fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"final_perm_importance_{region_name.replace(' ', '')}.png"), dpi=300)
    plt.show()
#------------------------------------------------------------------------------------------

# Partial dependence plots
for region_name, dfs in pdp_data.items():
    if not dfs:  # empty list
        print(f"No PDP data found for {region_name}, skipping PDP plot.")
        continue

    n_plots = len(dfs)
    fig, axes = plt.subplots(1, n_plots, figsize=(12, 4), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for i, df in enumerate(dfs[:3]): # use top 3 PDP files if more exist
        feature = df["feature"].unique()[0]

        axes[i].plot(df["feature_value"], df["partial_dependence"], lw=5)
        axes[i].set_xlabel(clean_feature_name(feature), fontsize=18)

        if i == 0:
            axes[i].set_ylabel("Partial dependence (SSS)", fontsize=16)
        else:
            axes[i].set_ylabel("")

        axes[i].tick_params(axis="both", which="major", labelsize=16)
        axes[i].grid(axis='both', linestyle='--', alpha=0.4)

    plt.suptitle(f"{region_name.replace('Var',' Var Region')}: PDP — Top 3",
                 y=0.93, fontsize=22, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"pdp_top3_{region_name}Region.png"), dpi=300)
    plt.show()

#------------------------------------------------------------------------------------------
# Model interpretability statistics 
# Feature importance stats
summary_rows = []

for region_name, df in region_dfs.items():
    df = df.copy()
    df["feature"] = df["feature"].apply(clean_feature_name)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    
    # Compute percentages
    df["percent_total"] = df["importance"] / df["importance"].sum() * 100
    top_feat = df.loc[0, "feature"]
    top_imp = df.loc[0, "importance"]
    top_pct = df.loc[0, "percent_total"]
    top3_pct = df.head(3)["percent_total"].sum()
    
    # Lagged vs total importance
    lagged_imp = df[df["feature"].str.contains("lag", case=False)]["importance"].sum()
    lagged_pct = 100 * lagged_imp / df["importance"].sum()
    
    summary_rows.append({
        "Region": region_name,
        "Top Feature": top_feat,
        "Top Importance": f"{top_imp:.3f}",
        "Top %": f"{top_pct:.1f}%",
        "Top3 %": f"{top3_pct:.1f}%",
        "Lagged %": f"{lagged_pct:.1f}%"
    })

importance_summary = pd.DataFrame(summary_rows)
# -------------------------------
# PDP stats (ΔSSS amplitude and correlation)
pdp_stats = []

for region_name, dfs in pdp_data.items():
    for df in dfs:
        feature = clean_feature_name(df["feature"].iloc[0])
        delta_sss = df["partial_dependence"].max() - df["partial_dependence"].min()
        corr = np.corrcoef(df["feature_value"], df["partial_dependence"])[0, 1]
        pdp_stats.append({
            "Region": region_name,
            "Feature": feature,
            "ΔSSS (PSU)": f"{delta_sss:.3f}",
            "Corr(feature, PDP)": f"{corr:.2f}"
        })

pdp_summary = pd.DataFrame(pdp_stats)
# -------------------------------
# Combine sumamry table (mean by region)
# Normalize region names in pdp_summary 
pdp_summary["Region"] = (
    pdp_summary["Region"]
    .replace({
        "LowVar": "Low Var Region",
        "MedVar": "Med Var Region",
        "HighVar": "High Var Region"
    })
)

# Clean Corr(feature, PDP) column
pdp_summary["Corr(feature, PDP)"] = (
    pdp_summary["Corr(feature, PDP)"]
    .astype(str)
    .str.extract(r"([-+]?\d*\.?\d+)")  
    .astype(float)
)

# Compute Mean ± SD ΔSSS per region 
mean_std_dsss = (
    pdp_summary.groupby("Region")["ΔSSS (PSU)"]
    .apply(lambda x: f"{x.astype(float).mean():.3f} ± {x.astype(float).std():.3f}")
    .reset_index()
    .rename(columns={"ΔSSS (PSU)": "Mean ± SD ΔSSS"})
)

# Merge with importance_summary to align on top feature
top_feature_corr = pd.merge(
    importance_summary[["Region", "Top Feature"]],
    pdp_summary[["Region", "Feature", "Corr(feature, PDP)"]],
    left_on=["Region", "Top Feature"],
    right_on=["Region", "Feature"],
    how="left"
).drop(columns="Feature")

top_feature_corr.rename(columns={"Corr(feature, PDP)": "Corr(Top Feature, PDP)"}, inplace=True)

# Combine everything
combined_importance_pdp_summary = (
    importance_summary
    .merge(mean_std_dsss, on="Region", how="left")
    .merge(top_feature_corr, on="Region", how="left")
)

#------------------------------------------------------------------------------------------
# Box plots of outer fold R2 by models
def clean_model_name(model):
    model = model.replace("RandomForest", "RF").replace("XGBoost", "XGB")
    model = model.replace("Baseline_Linear", "Baseline_LR")
    return model

# Map each region to its DataFrame
region_dfs = {
    "LowVar": nested_cv_all_models_LowVar_df,
    "MedVar": nested_cv_all_models_MedVar_df,
    "HighVar": nested_cv_all_models_HighVar_df
}

# Loop over each region and plot
for region_name, df in region_dfs.items():
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df, x="model", y="R2", hue="model",
        order=hue_order, hue_order=hue_order,
        palette=model_palette, dodge=False
    )

    plt.title(f"{region_name.replace('Var', ' Var Region')}: R² by Model",
              fontsize=21, fontweight='bold')
    plt.xlabel("Model", fontsize=18, fontweight='bold')
    plt.ylabel("R²", fontsize=18, fontweight='bold')

    ax = plt.gca()
    xticklabels = [clean_model_name(tick.get_text()) for tick in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels, fontsize=18, fontweight='bold')

    plt.yticks(fontsize=18)
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Save each figure
    out_path = os.path.join(OUT_DIR, f"model_r2_distribution_{region_name}Region.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

#------------------------------------------------------------------------------------------
# Q–Q plot
# Extract residuals
residuals = diag_data_foldFullData_LowVar_df["residual"]

# Q–Q plot of residuals
plt.figure(figsize=(5,5))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Low Var Region: Q–Q Plot of Residuals", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
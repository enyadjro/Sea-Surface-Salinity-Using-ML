'''
Create final statistics tables and presentation figures from EDA and Feature Engineering for 
the 3 sea surface salinity (SSS) regions: low, medium and high SSS variability
'''
# Clear variables and console
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic('reset', '-f')
get_ipython().run_line_magic('clear', '')

# Imports libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import os
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Paths
BASE_DIR = os.getcwd()
OUT_DIR = os.path.join("EDA_FE_figs_output")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "LowVar": "low_variables_revised.csv",
    "MedVar": "med_variables_revised.csv",
    "HighVar": "high_variables_revised.csv"
}

#------------------------------------------------------------------------------------------
# Load and preprocess data
DATA = {}
for key, fname in FILES.items():
    fpath = os.path.join(BASE_DIR, fname)
    if not os.path.exists(fpath):
        print(f" Missing file: {fname}. Skipping.")
        continue
    df = pd.read_csv(fpath)
    region = key.replace("Var", "").lower()
    prefix = f"{region}_"

    # Ensure numeric and sorted time
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year", "Month"]).sort_values(["Year", "Month"])

    # engineered fields 
    if f"{prefix}evap" in df.columns and f"{prefix}ppt" in df.columns:
        df[f"{prefix}E_minus_P"] = df[f"{prefix}evap"] - df[f"{prefix}ppt"]
    if all(c in df.columns for c in [f"{prefix}ucurr", f"{prefix}vcurr"]):
        df[f"{prefix}curr_mag"] = np.sqrt(df[f"{prefix}ucurr"]**2 + df[f"{prefix}vcurr"]**2)
    if all(c in df.columns for c in [f"{prefix}uwind", f"{prefix}vwind"]):
        df[f"{prefix}wind_mag"] = np.sqrt(df[f"{prefix}uwind"]**2 + df[f"{prefix}vwind"]**2)

    DATA[key] = df

#------------------------------------------------------------------------------------------
# Test whether SSS differs significantly between the 3 regimes
sss_samples = {}

for label, df in DATA.items():
    region = label.replace("Var", "").lower()
    col = f"{region}_sss"
    if col not in df.columns:
        print(f"SSS column {col} not found in {label}, skipping.")
        continue

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        print(f"No valid SSS data in {label}, skipping.")
        continue

    sss_samples[label] = s

# Quick summary 
for label, s in sss_samples.items():
    print(f"{label}: n = {len(s)}, mean SSS = {s.mean():.3f}, std = {s.std(ddof=1):.3f}")

# One-way ANOVA across the regimes
low_sss  = sss_samples["LowVar"]
med_sss  = sss_samples["MedVar"]
high_sss = sss_samples["HighVar"]

F_stat, p_anova = stats.f_oneway(low_sss, med_sss, high_sss)
print(f"\nOne-way ANOVA on SSS across regimes: F = {F_stat:.2f}, p = {p_anova:.3e}")

# Pairwise Welch t-tests with Bonferroni correction (3 comparisons)
def pairwise_test(a, b, name_a, name_b, m_corr=3):
    t, p_raw = stats.ttest_ind(a, b, equal_var=False)
    p_corr = min(p_raw * m_corr, 1.0)
    d_mean = a.mean() - b.mean()
    print(
        f"{name_a} vs {name_b}: Δmean = {d_mean:.3f} PSU, "
        f"t = {t:.2f}, p_raw = {p_raw:.3e}, p_Bonf = {p_corr:.3e}"
    )

print("\nPairwise Welch t-tests (with Bonferroni correction):")
pairwise_test(low_sss,  med_sss,  "LowVar",  "MedVar")
pairwise_test(low_sss,  high_sss, "LowVar",  "HighVar")
pairwise_test(med_sss,  high_sss, "MedVar",  "HighVar")

#------------------------------------------------------------------------------------------
# Descriptive stats computation and tables 
def compute_region_stats(region_label, df, out_dir):
    """
    Compute Mean, STD, Min, 25%, 50%, 75%, Max, Skewness
    for the main physical variables in one region and save to CSV.
    """
    region = region_label.replace("Var", "").lower()
    prefix = f"{region}_"

    # Base variables to report 
    base_vars = ["sss", "sst", "sla", "evap", "ppt", "runoff",
                 "uwind", "vwind", "ucurr", "vcurr"]

    # Names for the rows
    var_pretty = {
        "sss": "SSS",
        "sst": "SST",
        "sla": "SSH",   # rename SLA → SSH 
        "evap": "Evap",
        "ppt": "Ppt",
        "runoff": "Runoff",
        "uwind": "Uwind",
        "vwind": "Vwind",
        "ucurr": "Ucurr",
        "vcurr": "Vcurr",
    }

    rows = []
    for v in base_vars:
        col = f"{prefix}{v}"
        if col not in df.columns:
            # e.g. no runoff for low/med regions (open ocean)
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        desc = s.describe(percentiles=[0.25, 0.5, 0.75])
        rows.append({
            "Variable": var_pretty[v],
            "Mean":      desc["mean"],
            "STD":       desc["std"],
            "Min":       desc["min"],
            "25%":       desc["25%"],
            "50%":       desc["50%"],
            "75%":       desc["75%"],
            "Max":       desc["max"],
            "Skewness":  s.skew()
        })

    stats_df = pd.DataFrame(rows).set_index("Variable")

    # Save to csv
    csv_name = f"descriptive_stats_{region_label}.csv"
    csv_path = os.path.join(out_dir, csv_name)
    stats_df.to_csv(csv_path, float_format="%.4f")
    print(f"Saved descriptive stats table for {region_label} to {csv_path}")

    return stats_df

# compute and save stats for each region
STATS_TABLES = {}
for region_label, df in DATA.items():
    STATS_TABLES[region_label] = compute_region_stats(region_label, df, OUT_DIR)

#------------------------------------------------------------------------------------------
# Function: Time series anomaly Plot
def plot_anomaly(variable, units, title, out_filename, DATA, OUT_DIR):
    plt.figure(figsize=(14, 6))
    colors = {"LowVar": "tab:blue", "MedVar": "tab:red", "HighVar": "tab:green"}
    all_time_numeric = []

    for label, df in DATA.items():
        region = label.replace("Var", "").lower()
        prefix = f"{region}_"
        if f"{prefix}{variable}" not in df.columns:
            print(f"{variable.upper()} not found in {label}")
            continue

        time_numeric = df["Year"] + (df["Month"] - 1) / 12.0
        anomaly = df[f"{prefix}{variable}"] - df[f"{prefix}{variable}"].mean()
        plt.plot(time_numeric, anomaly, label=label.replace("Var", " Variability"),
                 linewidth=3.0, color=colors[label])
        all_time_numeric.append(time_numeric)

    ax = plt.gca()
    plt.title(title, fontsize=20, pad=12, fontweight='bold')
    ax.set_xlabel("Time (Year)", fontsize=16, labelpad=10)
    ax.set_ylabel(f"{variable.upper()} Anomaly ({units})", fontsize=16, labelpad=10)
    ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.7)

    tmin = float(np.floor(min(s.min() for s in all_time_numeric)))
    tmax = float(np.ceil(max(s.max() for s in all_time_numeric)))
    major_years = np.arange(int(tmin), int(tmax) + 1, 1)
    
    ax.set_xlim(tmin, tmax)
    ax.xaxis.set_major_locator(ticker.FixedLocator(major_years))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1/6.0))
    
    ax.tick_params(axis='both', which='major', direction='out', length=8, width=1.4, color='black', labelsize=16)
    ax.tick_params(axis='both', which='minor', direction='out', length=4, width=1.0, color='black')
    ax.grid(which="major", linestyle="--", alpha=0.3)
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 1.01), ncol=3, frameon=False, fontsize=16)
    plt.tight_layout()

    plt.savefig(os.path.join(OUT_DIR, out_filename), dpi=300, bbox_inches='tight')
    plt.show()

# Run 
plot_anomaly("sss", "PSU", "SSS Anomaly — Regional Comparison", "sss_anomaly_comparison_timeseries.png", DATA, OUT_DIR)
plot_anomaly("sst", "°C", "SST Anomaly — Regional Comparison", "sst_anomaly_comparison_timeseries.png", DATA, OUT_DIR)

#------------------------------------------------------------------------------------------
# Function: Monthly Boxplots
def plot_box_monthly(variable, region_label, units, DATA, OUT_DIR):
    df = DATA.get(region_label)
    if df is None:
        return
    region = region_label.replace("Var", "").lower()
    prefix = f"{region}_"
    var_col = f"{prefix}{variable}"

    if var_col not in df.columns:
        print(f" Column {var_col} not found in {region_label}")
        return

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Month", y=var_col, width=0.6, fliersize=4)
    ax = plt.gca()
    plt.title(f"{region_label.replace('Var', ' Variability Region')}: {variable.upper()}",
              fontsize=20, pad=8, fontweight='bold')
    
    ax.set_ylabel(f"{variable.upper()} ({units})", fontsize=18, labelpad=10)
    ax.set_xlabel("Month", fontsize=18, labelpad=6)
    
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(axis='y', linestyle="--", alpha=1)
    
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, f"box_month_{variable}_{region_label}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

for variable, units in [("sss", "PSU"), ("sst", "°C"), ("runoff", "m/month")]:
    for region_label in ["LowVar", "MedVar", "HighVar"]:
        plot_box_monthly(variable, region_label, units, DATA, OUT_DIR)

#------------------------------------------------------------------------------------------
# FUNCTION: Pairplots (engineered)
def make_engineered_pairplot(region_label, include_all, DATA, OUT_DIR):
    df = DATA.get(region_label)
    if df is None:
        return
    region = region_label.replace("Var", "").lower()
    prefix = f"{region}_"
    save_name = f"pairplot_engineered_{region_label}{'_ALL' if include_all else ''}.png"

    # Mapping 
    var_map_all = {
        f"{prefix}sss": "SSS", f"{prefix}sst": "SST", f"{prefix}sla": "SSH",
        f"{prefix}wind_mag": "Wind_mag", f"{prefix}curr_mag": "Curr_mag",
        f"{prefix}E_minus_P": "E-P", f"{prefix}runoff": "Runoff"
    }
    var_map_subset = {k: v for k, v in var_map_all.items() if v in ["SSS", "SST", "Wind_mag", "E-P", "Runoff"]}
    var_map = var_map_all if include_all else var_map_subset

    # Drop runoff for low/med
    if region in ["low", "med"]:
        var_map = {k: v for k, v in var_map.items() if "runoff" not in k}

    cols_present = [c for c in var_map.keys() if c in df.columns]
    if not cols_present:
        print(f"No valid columns for {region_label}")
        return

    df_plot = df[cols_present].rename(columns=var_map)
    sns.set_theme(style="whitegrid")
    alpha_val = 0.4 if include_all else 0.8
    font_size = 14 if include_all else 18

    g = sns.pairplot(df_plot, diag_kind="hist", corner=False,
                     plot_kws={"alpha": alpha_val, "s": 25, "color": "black", "edgecolor": "none"})
    for ax in g.axes.flatten():
        if ax:
            ax.set_xlabel(ax.get_xlabel(), fontsize=font_size, fontweight="bold")
            ax.set_ylabel(ax.get_ylabel(), fontsize=font_size, fontweight="bold")
            
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_name}")

for region_label in ["LowVar", "MedVar", "HighVar"]:
    make_engineered_pairplot(region_label, True, DATA, OUT_DIR)
    make_engineered_pairplot(region_label, False, DATA, OUT_DIR)

#------------------------------------------------------------------------------------------
# Funtion: Lead-Lag Correlation Heatmaps
def plot_lagcorr_heatmap(region_label, target_var, DATA, OUT_DIR, lags=range(-3,4)):
    df = DATA.get(region_label)
    if df is None:
        return
    region = region_label.replace("Var", "").lower()
    prefix = f"{region}_"
    save_name = f"lagcorr_{target_var.upper()}_{region_label}.png"
    target_map = {"ssh": f"{prefix}sla", "e-p": f"{prefix}E_minus_P"}
    target_col = target_map.get(target_var.lower(), f"{prefix}{target_var.lower()}")

    if target_col not in df.columns:
        print(f"Target {target_col} not found for {region_label}")
        return

    drivers = {
        "SST": f"{prefix}sst", "SSS": f"{prefix}sss", "E-P": f"{prefix}E_minus_P",
        "Curr_mag": f"{prefix}curr_mag", "SSH": f"{prefix}sla",
        "Wind_mag": f"{prefix}wind_mag", "Runoff": f"{prefix}runoff"
    }
    if region in ("low", "med"): drivers.pop("Runoff", None)
    drivers = {k:v for k,v in drivers.items() if v in df.columns and v != target_col}
    if not drivers:
        print(f"No valid drivers for {region_label}")
        return

    heatmap_data = pd.DataFrame(index=drivers.keys(), columns=lags, dtype=float)
    for disp, col in drivers.items():
        for lag in lags:
            pair = pd.concat([df[target_col], df[col].shift(lag)], axis=1).dropna()
            heatmap_data.loc[disp, lag] = pair.corr().iloc[0,1]

    plt.figure(figsize=(14,6))
    ax = sns.heatmap(heatmap_data, cmap="coolwarm", center=0, vmin=-0.8, vmax=0.8,
                     cbar_kws={'label':'Correlation'}, linewidths=0.5, linecolor='white')
    
    plt.title(f"{region_label.replace('Var',' Variability Region')}: {target_var.upper()}",
              fontsize=20, pad=12, fontweight='bold')
    ax.set_xlabel("Lag (months)", fontsize=18, labelpad=8)
    ax.set_ylabel("Driver Variable", fontsize=18, labelpad=8)
    
    plt.yticks(rotation=0, fontsize=16); plt.xticks(fontsize=16)
    ax.text(-0.1, len(drivers)+0.6, "Driver Leads", fontsize=16, ha='left', va='center')
    ax.text(len(lags)+0.1, len(drivers)+0.6, f"{target_var.upper()} Leads", fontsize=16, ha='right', va='center')
    
    for row in heatmap_data.index:
        vals = heatmap_data.loc[row].astype(float)
        if vals.abs().dropna().empty: continue
        best_lag = vals.abs().idxmax()
        y = list(heatmap_data.index).index(row) + 0.5
        x = list(lags).index(best_lag) + 0.5
        plt.scatter(x, y, marker='*', s=200, color='black')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_name}")

for region in ["LowVar", "MedVar", "HighVar"]:
    for target in ["sss", "sst", "ssh", "e-p", "wind_mag", "curr_mag", "runoff"]:
        if region in ("LowVar","MedVar") and target == "runoff": continue
        plot_lagcorr_heatmap(region, target, DATA, OUT_DIR)

#------------------------------------------------------------------------------------------
# Function: PCA Analysis
def run_pca_for_region(region_label, DATA, OUT_DIR):
    df = DATA.get(region_label)
    if df is None:
        return
    
    region = region_label.lower().replace("var", "")
    prefix = f"{region}_"
    features = [f"{prefix}sss", f"{prefix}sst", f"{prefix}sla",
                f"{prefix}wind_mag", f"{prefix}curr_mag", f"{prefix}E_minus_P"]
    if region == "high": features.append(f"{prefix}runoff")

    df_pca = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pca)
    pca = PCA().fit(X_scaled)
    explained_var = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_var)+1), explained_var, marker="o", linewidth=3)
    plt.xticks(range(1, len(explained_var)+1))
    
    plt.xlabel("Number of Components", fontsize=16)
    plt.ylabel("Cumulative Explained Variance", fontsize=16)
    plt.title(f"{region_label.capitalize()} Variability Region: PCA Explained Variance",
              fontsize=20, pad=10, fontweight='bold')
    
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(OUT_DIR, f"pca_variance_{region_label}_FINAL.png"), dpi=300)
    plt.show()

    pca_limited = PCA(n_components=3).fit(X_scaled)
    loadings = pd.DataFrame(pca_limited.components_.T, index=features, columns=["PC1","PC2","PC3"])
    rename_map = {f"{prefix}sss":"SSS", f"{prefix}sst":"SST", f"{prefix}sla":"SSH",
                  f"{prefix}wind_mag":"Wind_mag", f"{prefix}curr_mag":"Curr_mag",
                  f"{prefix}E_minus_P":"E-P", f"{prefix}runoff":"Runoff"}
    
    loadings.index = loadings.index.map(rename_map)
    fig, ax = plt.subplots(figsize=(12,6))
    sns.set_style("whitegrid")
    bar_container = loadings.plot(kind="bar", width=0.8, ax=ax)
    
    for pc_index, pc in enumerate(["PC1","PC2","PC3"]):
        values = loadings[pc]
        best_idx = np.nanargmax(np.abs(values))
        bar = ax.containers[pc_index].patches[best_idx]
        x_star = bar.get_x() + bar.get_width()/2
        y_star = bar.get_height()
        ax.scatter(x_star, y_star, marker="*", s=350, color="black", zorder=10)
        
    ax.set_title(f"{region_label.capitalize()} Variability Region - PCA Loadings",
                 fontsize=20, fontweight='bold')
    ax.set_xlabel("Variable", fontsize=16)
    ax.set_ylabel("Loading Strength", fontsize=16)
    
    plt.xticks(rotation=0, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(title="Principal Component", fontsize=11, title_fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"pca_loadings_{region_label}_FINAL.png"), dpi=300)
    plt.show()

for region in ["LowVar", "MedVar", "HighVar"]:
    run_pca_for_region(region, DATA, OUT_DIR)

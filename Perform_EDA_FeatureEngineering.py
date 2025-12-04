"""
Perform EDA and Feature Engineering for the 3 sea surface salinity (SSS) regions:
low, medium and high SSS variability
"""
# Clear variables and console
from IPython import get_ipython
ipython = get_ipython()
if ipython:
    ipython.run_line_magic('reset', '-f')
    ipython.run_line_magic('clear', '')

# Import libraries
import os, warnings, calendar
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

# Paths and files
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR)

# List of input files
ORIGINAL_FILES = ["low_variables.csv", "med_variables.csv", "high_variables.csv"]

# Options
DO_PCA = True
N_PCA_COMPONENTS = 5
FIG_DPI = 180

# Lagged correlation settings
MAX_LAG = 3    # ±3 months
LAG_RANGE = range(-MAX_LAG, MAX_LAG + 1)
LAG_METRIC_ABS = True   # pick peak by absolute correlation magnitude
LAG_HEATMAP_CMAP = "coolwarm"
LAG_HEATMAP_CENTER = 0.0
#------------------------------------------------------------------------------------------
# Helpers
def iqr_bounds(series: pd.Series):
    """
    Compute inter-quartile range
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return q1, q3, iqr, lb, ub

def compute_lagged_correlations(df, target_col, driver_cols, lags, abs_metric=True):
    """
    Compute lagged correlations of each driver vs target over lags.
    Returns:
      heatmap_df: DataFrame (rows=drivers, cols=lags, values=signed correlation)
      best_rows: list of dicts {Driver, Best_Lag, Peak_Correlation}
    """
    results = []
    for var in driver_cols:
        for lag in lags:
            r = df[target_col].corr(df[var].shift(lag))
            results.append({"Driver": var, "Lag": lag, "Correlation": r})

    df_lcorr = pd.DataFrame(results)
    heatmap_df = df_lcorr.pivot(index="Driver", columns="Lag", values="Correlation")

    best_rows = []
    for drv in heatmap_df.index:
        series = heatmap_df.loc[drv]
        svalid = series.dropna()
        if svalid.empty:
            best_rows.append({"Driver": drv, "Best_Lag": None, "Peak_Correlation": None})
            continue
        if abs_metric:
            idx = svalid.abs().idxmax()
        else:
            idx = svalid.idxmax()
        best_rows.append({"Driver": drv, "Best_Lag": int(idx), "Peak_Correlation": float(series.loc[idx])})

    return heatmap_df, best_rows

def plot_lag_heatmap(heatmap_df, title, out_path, center=0.0, cmap="coolwarm", dpi=180):
    """
    Plot lead-lag heat maps
    """
    plt.figure(figsize=(10, max(4, 0.4 * len(heatmap_df.index))))
    ax = sns.heatmap(
        heatmap_df,
        cmap=cmap,
        center=center,
        annot=False,
        linewidths=0.5,
        linecolor='white'
    )
    ax.set_title(title)
    ax.set_xlabel("Lag (months) (negative = driver leads, positive = target leads)")
    ax.set_ylabel("Driver Variable")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
#------------------------------------------------------------------------------------------

# Convert m/s to m/month and save the *_revised.csv
revised_files = []
for FILE_NAME in ORIGINAL_FILES:
    full_path = os.path.join(DATA_DIR, FILE_NAME)
    if not os.path.exists(full_path):
        print(f"Missing file: {full_path} (skipping conversion)")
        continue

    region_tag = os.path.splitext(FILE_NAME)[0].split("_")[0]  
    print(f"\nConverting units for: {FILE_NAME}")

    # Load the files
    df_raw = pd.read_csv(full_path)
    assert "Year" in df_raw.columns and "Month" in df_raw.columns, "Expected 'Year' and 'Month' columns."

    # Types
    df_raw["Year"] = pd.to_numeric(df_raw["Year"], errors="coerce").astype("Int64")
    df_raw["Month"] = pd.to_numeric(df_raw["Month"], errors="coerce").astype("Int64")
    df_raw = df_raw.dropna(subset=["Year", "Month"]).copy()
    df_raw["Year"] = df_raw["Year"].astype(int)
    df_raw["Month"] = df_raw["Month"].astype(int)

    # Detect prefix (e.g., 'low_', 'med_', 'high_')
    candidate_cols = [c for c in df_raw.columns if "_" in c and c not in ["Year", "Month"]]
    prefix = (pd.Series([c.split("_")[0] for c in candidate_cols]).mode()[0] + "_") if candidate_cols else region_tag + "_"

    # Compute days in each month for each row (handles leap years)
    dt = pd.to_datetime(dict(year=df_raw["Year"], month=df_raw["Month"], day=1))
    days_in_month = dt.dt.daysinmonth
    seconds_per_day = 24 * 60 * 60
    factor = seconds_per_day * days_in_month # multiply m/s by this to get the m/month

    # Columns to convert
    col_evap = prefix + "evap"
    col_ppt = prefix + "ppt"
    col_runoff = prefix + "runoff"

    for col in [col_evap, col_ppt, col_runoff]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
            df_raw[col] = df_raw[col] * factor
        else:
            print(f"Column not found for conversion (skipping): {col}")

    # Save the revised data file
    revised_name = f"{region_tag}_variables_revised.csv"
    revised_path = os.path.join(DATA_DIR, revised_name)
    df_raw.to_csv(revised_path, index=False)
    revised_files.append(revised_name)
    print(f"Saved revised file: {revised_path}")

# If nothing is revised, stop
if not revised_files:
    raise FileNotFoundError("No revised files were created. Check original file paths and names.")
#------------------------------------------------------------------------------------------

# Full Pipeline using only the revised files
FILES = revised_files[:]  

for FILE_NAME in FILES:
    full_path = os.path.join(DATA_DIR, FILE_NAME)
    if not os.path.exists(full_path):
        print(f"Missing revised file: {full_path} (skipping)")
        continue

    region_tag = os.path.splitext(FILE_NAME)[0].split("_")[0]  
    OUT_DIR = os.path.join(DATA_DIR, "EDA_FE_outputs", region_tag)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nProcessing (REVISED): {FILE_NAME} → {OUT_DIR}")

    # -----------------------
    # Load and inspect
    df = pd.read_csv(full_path)
    assert "Year" in df.columns and "Month" in df.columns, "Expected 'Year' and 'Month' columns."

    # Detect column prefix (e.g., 'low_', 'med_', 'high_')
    candidate_cols = [c for c in df.columns if "_" in c and c not in ["Year", "Month"]]
    prefix = (pd.Series([c.split("_")[0] for c in candidate_cols]).mode()[0] + "_") if candidate_cols else region_tag + "_"
    strip = lambda c: c[len(prefix):] if c.startswith(prefix) else c

    # Types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Year", "Month"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    # Initial value columns (prefixed)
    value_cols = [c for c in df.columns if c.startswith(prefix)]
    print(f"Detected prefix: {prefix} | initial prefixed cols: {len(value_cols)}")

    # -----------------------
    # Descriptive stats 
    if len(value_cols) > 0:
        desc = df[value_cols].describe().T
        try:
            desc["skewness"] = df[value_cols].apply(skew, nan_policy="omit")
            desc["kurtosis"] = df[value_cols].apply(kurtosis, nan_policy="omit")
        except Exception:
            pass
        desc.index = [strip(c) for c in desc.index]
        desc.to_csv(os.path.join(OUT_DIR, f"{region_tag}_descriptive_stats.csv"))

    # -----------------------
    # Feature Engineering
    def get(base):
        name = prefix + base
        return name if name in df.columns else None

    # Base columns (may or may not exist in every file)
    sss, sst, sla = get("sss"), get("sst"), get("sla")
    uwind, vwind = get("uwind"), get("vwind")
    ucurr, vcurr = get("ucurr"), get("vcurr")
    ppt, evap = get("ppt"), get("evap")
    runoff = get("runoff")

    # Magnitudes
    if uwind and vwind:
        df[f"{prefix}wind_mag"] = np.sqrt(df[uwind]**2 + df[vwind]**2)
    if ucurr and vcurr:
        df[f"{prefix}curr_mag"] = np.sqrt(df[ucurr]**2 + df[vcurr]**2)

    # Freshwater forcing (E_minus_P) in m/month
    if ppt and evap:
        df[f"{prefix}E_minus_P"] = df[evap] - df[ppt]

    # Seasonality
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)

    # Lags for SSS/SST if present
    for base in ["sss", "sst"]:
        c = get(base)
        if c:
            df[f"{c}_lag1"] = df[c].shift(1)
            df[f"{c}_lag2"] = df[c].shift(2)

    # Interaction example
    if sst and f"{prefix}wind_mag" in df.columns:
        df[f"{prefix}sst_x_windmag"] = df[sst] * df[f"{prefix}wind_mag"]

    # Refresh value_cols after engineering
    value_cols = [c for c in df.columns if c.startswith(prefix)]

    # -----------------------
    # Visuals (distributions, Boxplots by month, time series)
    for col in value_cols:
        # Distributions
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution: {strip(col)}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"dist_{strip(col)}.png"), dpi=FIG_DPI)
        plt.close(fig)

        # Monthly boxplots
        fig, ax = plt.subplots(figsize=(7,4))
        sns.boxplot(data=df, x="Month", y=col, ax=ax)
        ax.set_title(f"Monthly Boxplot: {strip(col)}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"box_month_{strip(col)}.png"), dpi=FIG_DPI)
        plt.close(fig)

    # Time series for all prefixed variables (engineered + the raw)
    df_sorted = df.sort_values(["Year", "Month"]).copy()
    tindex = pd.to_datetime(dict(year=df_sorted["Year"], month=df_sorted["Month"], day=1))
    for col in value_cols:
        fig, ax = plt.subplots(figsize=(8,3.8))
        ax.plot(tindex, df_sorted[col], linewidth=1.2)
        ax.set_title(f"Time Series: {strip(col)}")
        ax.set_xlabel("Time"); ax.set_ylabel(strip(col))
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"ts_{strip(col)}.png"), dpi=FIG_DPI)
        plt.close(fig)
    # -----------------------
    
    # Monthly Climatology (Mean ± Std)
    if len(value_cols) > 0:
        clim = df.groupby("Month")[value_cols].agg(['mean', 'std'])
        clim.columns = ['_'.join(c) for c in clim.columns]  #flatten the data

        key_vars = ['sss', 'sst', 'sla', 'wind_mag', 'curr_mag', 'E_minus_P', 'runoff']
        for v in key_vars:
            col = f"{prefix}{v}"
            mcol, scol = f"{col}_mean", f"{col}_std"
            if mcol in clim.columns:
                fig, ax = plt.subplots(figsize=(7,4))
                months = range(1,13)
                ax.errorbar(months, clim[mcol], yerr=clim[scol], marker='o', linewidth=1.8, capsize=4)
                ax.set_xticks(months); ax.set_xticklabels(calendar.month_abbr[1:13])
                ax.set_title(f"Monthly Climatology of {strip(col)} (Mean ± Std)")
                ax.set_xlabel("Month"); ax.set_ylabel(strip(col))
                ax.grid(alpha=0.3, linestyle="--")
                fig.tight_layout()
                fig.savefig(os.path.join(OUT_DIR, f"{region_tag}_monthly_climatology_{v}.png"), dpi=FIG_DPI)
                plt.close(fig)
    # -----------------------
    
    # Pairplot (engineered variables only)
    engineered_list = []
    for base in ['sss', 'sst', 'sla', 'wind_mag', 'curr_mag', 'E_minus_P', 'runoff']:
        col = f"{prefix}{base}"
        if col in df.columns:
            engineered_list.append(col)

    if len(engineered_list) >= 2:
        df_pair = df[engineered_list].copy()
        df_pair.columns = [strip(c) for c in df_pair.columns]
        g = sns.pairplot(df_pair, diag_kind="hist",
                         plot_kws={'alpha':0.35, 'edgecolor':'none', 's':22})
        g.fig.suptitle(f"Pairplot of Engineered Drivers ({region_tag.capitalize()} Region, E−P)", y=1.02)
        g.savefig(os.path.join(OUT_DIR, f"{region_tag}_pairplot_engineered.png"), dpi=FIG_DPI)
        plt.close(g.fig)
    # -----------------------
    
    # Traditional Correlation (Lag 0) — engineered only
    if len(engineered_list) >= 2:
        corr_eng = df[engineered_list].corr(method="pearson")
        corr_eng.index = [strip(c) for c in corr_eng.index]
        corr_eng.columns = [strip(c) for c in corr_eng.columns]

        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr_eng, cmap="coolwarm", annot=False, ax=ax, square=True, center=0)
        ax.set_title("Correlation Heatmap (Lag 0) — Engineered Variables")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"{region_tag}_corr_heatmap_engineered.png"), dpi=FIG_DPI)
        plt.close(fig)

        corr_eng.to_csv(os.path.join(OUT_DIR, f"{region_tag}_correlations_engineered.csv"))
    # -----------------------
    
    # IQR Outliers: summary + flags 
    out_rows = []
    for col in value_cols:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        q1, q3, iqr, lb, ub = iqr_bounds(s)
        is_out = (df[col] < lb) | (df[col] > ub)
        out_rows.append({
            "Variable": strip(col), "Q1": q1, "Q3": q3, "IQR": iqr,
            "Lower_Bound": lb, "Upper_Bound": ub,
            "Outlier_Count": int(is_out.sum()),
            "Outlier_Pct": 100.0 * is_out.sum() / len(df)
        })
        df[f"{col}_is_outlier"] = is_out.astype(int)
    if len(out_rows) > 0:
        pd.DataFrame(out_rows).to_csv(os.path.join(OUT_DIR, f"{region_tag}_iqr_outliers_summary.csv"), index=False)
    # -----------------------
    
    #for PCA analysis 
    if DO_PCA:
        drop_cols = ["Year", "Month"]
        pca_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).select_dtypes(include=[np.number]).copy()
        
        # drop cols with >10% NaN then mean-impute
        thresh = int(0.9 * len(pca_df))
        pca_df = pca_df.dropna(axis=1, thresh=thresh).fillna(pca_df.mean())

        scaler = StandardScaler()
        Xs = scaler.fit_transform(pca_df.values)

        n_comp = min(N_PCA_COMPONENTS, Xs.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        Xp = pca.fit_transform(Xs)

        # explained variance
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(np.arange(1, n_comp+1), np.cumsum(pca.explained_variance_ratio_), marker="o")
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("PCA Explained Variance")
        fig.tight_layout()
        ev_path = os.path.join(OUT_DIR, f"{region_tag}_pca_explained_variance.png")
        fig.savefig(ev_path, dpi=FIG_DPI); plt.close(fig)

        # Loadings
        loadings = pd.DataFrame(pca.components_.T,
                                index=pca_df.columns,
                                columns=[f"PC{i+1}" for i in range(n_comp)])
        loadings_path = os.path.join(OUT_DIR, f"{region_tag}_pca_loadings.csv")
        loadings.to_csv(loadings_path)
    # -----------------------
    
    # Lagged Correlation Heatmaps (±3 months) for each engineered variable as Ttarget
    # Engineered set for lag analysis 
    lag_vars = [c for c in [f"{prefix}sss", f"{prefix}sst", f"{prefix}sla",
                            f"{prefix}wind_mag", f"{prefix}curr_mag",
                            f"{prefix}E_minus_P", f"{prefix}runoff"] if c in df.columns]

    best_rows_all_targets = []
    for target in lag_vars:
        drivers = [v for v in lag_vars if v != target]
        if len(drivers) == 0:
            continue

        heatmap_df, best_rows = compute_lagged_correlations(
            df, target_col=target, driver_cols=drivers, lags=LAG_RANGE, abs_metric=LAG_METRIC_ABS
        )

        # Re-label rows with stripped names; keep columns numeric-ordered
        heatmap_df.index = [strip(ix) for ix in heatmap_df.index]
        heatmap_df = heatmap_df[sorted(heatmap_df.columns)]

        target_name = strip(target)
        out_png = os.path.join(OUT_DIR, f"{region_tag}_lag_corr_{target_name}_heatmap.png")
        plot_lag_heatmap(
            heatmap_df,
            title=f"Lagged Correlation Heatmap with Target = {target_name} ({region_tag.capitalize()} Region)",
            out_path=out_png,
            center=LAG_HEATMAP_CENTER,
            cmap=LAG_HEATMAP_CMAP,
            dpi=FIG_DPI
        )

        for row in best_rows:
            row["Target"] = target_name
            row["Driver"] = strip(row["Driver"])
            best_rows_all_targets.append(row)

    if len(best_rows_all_targets) > 0:
        best_df = pd.DataFrame(best_rows_all_targets)[["Target", "Driver", "Best_Lag", "Peak_Correlation"]]
        best_df.sort_values(by=["Target", "Driver"], inplace=True)
        best_df.to_csv(os.path.join(OUT_DIR, f"{region_tag}_best_lag_summary.csv"), index=False)
    # -----------------------
    
    # Model-ready features export
    model_cols = ["Year", "Month"]
    for base in ["sss", "sst", "sla", "wind_mag", "curr_mag", "E_minus_P", "runoff",
                 "sss_lag1", "sss_lag2", "sst_lag1", "sst_lag2", "month_sin", "month_cos"]:
        col = (f"{prefix}{base}" if "lag" not in base and base not in ["month_sin", "month_cos"]
               else (f"{prefix}sss_lag1" if base=="sss_lag1" else
                     f"{prefix}sss_lag2" if base=="sss_lag2" else
                     f"{prefix}sst_lag1" if base=="sst_lag1" else
                     f"{prefix}sst_lag2" if base=="sst_lag2" else base))
        if col in df.columns:
            model_cols.append(col)

    model_cols = [c for c in model_cols if c in df.columns]
    if len(model_cols) > 0:
        model_df = df[model_cols].copy()
        model_path = os.path.join(OUT_DIR, f"{region_tag}_model_features.csv")
        model_df.to_csv(model_path, index=False)
    #------------------------------------------------------------------------------------------
    
    # Save engineered + flags dataset
    df.to_csv(os.path.join(OUT_DIR, f"{region_tag}_engineered_flagged.csv"), index=False)



'''
Build and Evaluate models for predicting sea surface salinity (SSS) in 3 regions of varying variability:
low, medium and high SSS variability
'''
# Clear variables and console
from IPython import get_ipython
ipython = get_ipython()
if ipython:
    ipython.run_line_magic('reset', '-f')
    ipython.run_line_magic('clear', '')

# Import libraries
import os
os.environ["NUMEXPR_MAX_THREADS"] = "8"  
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["NUMPY_WARN_IF_OVERFLOW"] = "0"
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.seterr(all='ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from typing import Dict, List

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.stats import pearsonr

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#------------------------------------------------------------------------------------------
# Config
BASE_DIR = os.getcwd()
OUT_ROOT = os.path.join(BASE_DIR, "model_sss_region_outputs")
os.makedirs(OUT_ROOT, exist_ok=True)

REGION_CONFIG = [
    dict(name="low",  csv=os.path.join(BASE_DIR, "low_variables_revised.csv"),  prefix="low"),
    dict(name="med",  csv=os.path.join(BASE_DIR, "med_variables_revised.csv"),  prefix="med"),
    dict(name="high", csv=os.path.join(BASE_DIR, "high_variables_revised.csv"), prefix="high"),
]

# Modeling knobs
MAX_LAG     = 3  
N_OUTER     = 6
N_INNER     = 4
RSEED       = 42
USE_HARMONICS = True  # True = add month_sin/cos, False = deseasonalize (Option B)
#------------------------------------------------------------------------------------------
# Helper structures
@dataclass
class RegionTags:
    sss: str
    sst: str
    evap: str
    ppt: str
    ucurr: str
    vcurr: str
    uwind: str
    vwind: str
    runoff: str
    ssh: str

def build_tags(prefix: str) -> RegionTags:
    return RegionTags(
        sss=f"{prefix}_sss",
        sst=f"{prefix}_sst",
        evap=f"{prefix}_evap",
        ppt=f"{prefix}_ppt",
        ucurr=f"{prefix}_ucurr",
        vcurr=f"{prefix}_vcurr",
        uwind=f"{prefix}_uwind",
        vwind=f"{prefix}_vwind",
        runoff=f"{prefix}_runoff",
        ssh=f"{prefix}_sla",
    )

def deseasonalize_monthly(df: pd.DataFrame, cols: List[str], month_col="Month") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        clim = out.groupby(month_col)[c].transform("mean")
        out[c] = out[c] - clim
    return out

#------------------------------------------------------------------------------------------
# Core runner per region
def run_region(region_name: str, csv_path: str, prefix: str) -> Dict:
    out_dir = os.path.join(OUT_ROOT, region_name)
    os.makedirs(out_dir, exist_ok=True)

    tags = build_tags(prefix)
    TARGET_COL = tags.sss

    df = pd.read_csv(csv_path)
    if "Year" in df.columns:
        df = df.drop(columns=["Year"])

    # Seasonality handling
    if USE_HARMONICS:
        if "Month" in df.columns:
            df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
            df = df.drop(columns=["Month"])
        month_features = ["month_sin", "month_cos"]
    else:
        deseasonal_cols = [
            tags.sss, tags.sst, tags.evap, tags.ppt, tags.ucurr, tags.vcurr,
            tags.uwind, tags.vwind, tags.runoff, tags.ssh
        ]
        df = deseasonalize_monthly(df, deseasonal_cols, month_col="Month")
        month_features = []

    # Derived features
    df["curr_mag"] = np.sqrt(df[tags.ucurr]**2 + df[tags.vcurr]**2)
    df["wind_mag"] = np.sqrt(df[tags.uwind]**2 + df[tags.vwind]**2)
    df["EPR"] = (df[tags.evap] - df[tags.ppt]) + df[tags.runoff]

    # Drop variables
    df = df.drop(columns=[tags.evap, tags.ppt, tags.ucurr, tags.vcurr,
                          tags.uwind, tags.vwind, tags.runoff])

    base_features = [tags.sst, "EPR", "curr_mag", "wind_mag", tags.ssh] + month_features

    for col in [tags.sst, "EPR", "curr_mag", "wind_mag", tags.ssh]:
        for lag in range(1, MAX_LAG + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df = df.dropna().reset_index(drop=True)
    X_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[X_cols].copy()
    y = df[TARGET_COL].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = np.clip(X, -1e6, 1e6)
    X = pd.DataFrame(X, columns=X_cols)

    # Drop columns that are entirely zeros (e.g.,runoff in low and med variability regions)
    zero_cols = [c for c in X.columns if (X[c] == 0).all()]
    if zero_cols:
        print(f"[{region_name}] Dropped {len(zero_cols)} all-zero columns: {zero_cols[:5]}{'...' if len(zero_cols)>5 else ''}")
    
    # Feature skew report 
    try:
        present_bases = [c for c in base_features if c in X.columns]
        pd.Series(X[present_bases].skew()).sort_values(ascending=False).to_csv(
            os.path.join(out_dir, "feature_skew_report.csv")
        )
    except Exception:
        pass

    numeric_features = X.columns.tolist()

    # Preprocess    
    preprocess = ColumnTransformer(
        transformers=[("num", Pipeline([
            ("yeo", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("scaler", StandardScaler(with_mean=True, with_std=True))
        ]), numeric_features)],
        remainder="drop")

    # Clipping helpers 
    def _clip_frame(df: pd.DataFrame, lo=-1e2, hi=1e2) -> pd.DataFrame:
        arr = np.clip(df.values, lo, hi)
        # Replace any remaining NaN/Inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=hi, neginf=lo)
        return pd.DataFrame(arr, columns=df.columns, index=df.index)
    
    def _clip_series(s: pd.Series, lo=-1e3, hi=1e3) -> pd.Series:
        arr = np.clip(s.values, lo, hi)
        arr = np.nan_to_num(arr, nan=0.0, posinf=hi, neginf=lo)
        return pd.Series(arr, index=s.index, name=s.name)
    
        # Overflow diagnostic helper 
    def check_extreme_values(df, threshold=1e5, label=""):
        big_vals = (np.abs(df) > threshold).sum()
        if big_vals.any():
            offenders = big_vals[big_vals > 0]
            print(f"[{label}] Extreme values detected in {len(offenders)} columns:")
            print(offenders.sort_values(ascending=False).head(10))
#------------------------------------------------------------------------------------------

    # Hyperparameter Grids
    models_and_grids = {
        "Baseline_Linear": (LinearRegression(), {}),

        "ElasticNet": (
            ElasticNet(max_iter=200000, random_state=RSEED, tol=1e-4, selection="random"),
            {
                "model__alpha": np.logspace(-4, 2, 12),   
                "model__l1_ratio": np.linspace(0.05, 0.95, 19)  
            }
        ),

        "SVR": (
            SVR(),
            {
                "model__C":[0.1, 1, 10, 100],       
                "model__epsilon": [0.05, 0.1, 0.2],
                "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                "model__kernel": ["rbf"],
                "model__degree": [2, 3]

            }
        ),

        "RandomForest": (
            RandomForestRegressor(random_state=RSEED, n_jobs=-1),
            {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [None, 6, 10, 15],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
                "model__bootstrap": [True, False],
            }
        ),

        "XGBoost": (
            XGBRegressor(
                random_state=RSEED,
                objective="reg:squarederror",
                n_estimators=600,
                tree_method="auto",
                n_jobs=-1
            ),
            {
                "model__max_depth": [3, 5, 7, 10],
                "model__learning_rate": [0.01, 0.03, 0.1, 0.2],
                "model__subsample": [0.6, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__gamma": [0, 0.1, 0.5, 1],
                "model__reg_alpha": [0, 0.01, 0.1],
                "model__reg_lambda": [1, 1.5, 2.0],
                "model__n_estimators": [400, 800]

            }
        ),
    }

    outer_cv = TimeSeriesSplit(n_splits=N_OUTER)
    inner_cv = TimeSeriesSplit(n_splits=N_INNER)
    #------------------------------------------------------------------------------------------
    
    # Diagnostic plot function
    def plot_diagnostics(y_true, y_pred, model_name, fold_id):
        resid = y_true - y_pred
        
        # Handle constant arrays for Pearson r
        yt = np.asarray(y_true).ravel()
        yp = np.asarray(y_pred).ravel()
        
        var_ytrue = np.nanvar(yt)
        var_ypred = np.nanvar(yp)
        
        if len(yt) < 3 or var_ytrue == 0 or var_ypred == 0:
            r_value = np.nan
            reason = ("too few points" if len(yt) < 3 else
                      "constant y_true" if var_ytrue == 0 else
                      "constant y_pred")
            print(f"[{region_name} | Fold {fold_id} | {model_name}] Pearson r undefined "
                  f"({reason}). std(y_true)={np.sqrt(var_ytrue):.4g}, std(y_pred)={np.sqrt(var_ypred):.4g}")
        else:
            r_value, _ = pearsonr(yt, yp)
        
        r_label = f"r = {r_value:.3f}" if np.isfinite(r_value) else "r undefined"

        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
        sns.scatterplot(ax=axes[0], x=y_true, y=y_pred, alpha=0.7)
        lo, hi = y_true.min(), y_true.max()
        axes[0].plot([lo, hi], [lo, hi], "r--")
        axes[0].set_title(f"Fold {fold_id}: Predicted vs Actual ({model_name})")
        axes[0].text(0.05, 0.9, r_label,
                     transform=axes[0].transAxes, fontsize=12,
                     color="darkblue", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

        sns.scatterplot(ax=axes[1], x=y_pred, y=resid, alpha=0.7)
        axes[1].axhline(0, color="r", linestyle="--")
        axes[1].set_title(f"Fold {fold_id}: Residuals vs Predicted ({model_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"diag_fold{fold_id}_{model_name}.png"), dpi=300)
        plt.close()
    
        # Save the data used for this plot
        diag_df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": resid
        })
        diag_df.to_csv(os.path.join(out_dir, f"diag_data_fold{fold_id}_{model_name}.csv"), index=False)
    #------------------------------------------------------------------------------------------
    # Nested CV
    records_all, records_best = [], []
    fold_id = 0
    for tr, te in outer_cv.split(X):
        fold_id += 1
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        fold_results = []
        
        # Clip train/test split before any fitting 
        Xtr = _clip_frame(Xtr);  Xte = _clip_frame(Xte)
        ytr = _clip_series(ytr); yte = _clip_series(yte)
    #------------------------------------------------------------------------------------------

        for name, (est, grid) in models_and_grids.items():
            pipe = Pipeline([("prep", preprocess), ("model", est)])
            search = GridSearchCV(
                pipe,
                param_grid=grid,
                cv=inner_cv,
                scoring="r2",
                n_jobs=-1,
                refit=True
            )

            try:
                if not np.isfinite(Xtr.values).all():
                    print(f"[{region_name} | Fold {fold_id}] Non-finite values found after clipping.")
                    Xtr = _clip_frame(Xtr)
                
                check_extreme_values(Xtr, threshold=1e5, label=f"{region_name} | Fold {fold_id} | {name} | Train")

                search.fit(Xtr, ytr)
                inner = search.best_score_
                yhat = search.best_estimator_.predict(Xte)
                r2 = r2_score(yte, yhat)
                rmse = mean_squared_error(yte, yhat, squared=False)
                mae = mean_absolute_error(yte, yhat)
                fold_results.append(dict(
                    fold=fold_id,
                    model=name,
                    best_inner_R2=round(inner, 4),
                    R2=round(r2, 4),
                    RMSE=round(rmse, 4),
                    MAE=round(mae, 4),
                    best_estimator=search.best_estimator_  # Store the tuned model
                ))

            except Exception as e:
                print(f"[{region_name} | Fold {fold_id}] {name} failed: {e}")

        records_all.extend(fold_results)
        if fold_results:
            df_fold = pd.DataFrame(fold_results)
            
            # Select best model by outer test R2 (and use RMSE/MAE as tiebreakers)
            best_row = df_fold.sort_values(by=["R2", "RMSE", "MAE"], ascending=[False, True, True]).iloc[0]
            best_name = best_row["model"]
            
            # Get predictions for the already-trained best model
            best_idx = next(i for i, r in enumerate(fold_results) if r["model"] == best_name)
            best_r2 = fold_results[best_idx]["R2"]
            
            # Use tuned model directly (no extra refit)
            best_obj = next(r["best_estimator"] for r in fold_results if r["model"] == best_name)
            yhat_best = best_obj.predict(Xte)
            
            plot_diagnostics(yte, yhat_best, best_name, fold_id)
            records_best.append(best_row)
    #------------------------------------------------------------------------------------------
    
    # Save and Summaries
    all_results = pd.DataFrame(records_all)
    all_results.to_csv(os.path.join(out_dir, "nested_cv_all_models.csv"), index=False)
    by_model_all = all_results.groupby("model")[["R2","RMSE","MAE"]].mean().sort_values("R2", ascending=False)
    by_model_all.reset_index().rename(columns={"index": "model"}).to_csv(
    os.path.join(out_dir, "all_models_summary.csv"), index=False)

    best_per_fold = pd.DataFrame(records_best)
    best_per_fold.to_csv(os.path.join(out_dir, "nested_cv_best_per_fold_outerR2.csv"), index=False)

    by_model = best_per_fold.groupby("model")[["R2","RMSE","MAE"]].mean().sort_values("R2", ascending=False)
    by_model.reset_index().rename(columns={"index": "model"}).to_csv(
    os.path.join(out_dir, "best_model_summary.csv"), index=False)

    plt.figure(figsize=(8,5))
    sns.boxplot(data=all_results, x="model", y="R2", palette="coolwarm")
    plt.title(f"{region_name.upper()}: Outer-Fold R² by Model")
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_r2_distribution.png"), dpi=300)
    plt.close()

    # Use all_models_summary instead of best_per_fold summary
    best_overall_model = by_model_all.index[0]

    # Final refit
    est, grid = models_and_grids[best_overall_model]
    pipe_full = Pipeline([("prep", preprocess), ("model", est)])
    
    final_cv = TimeSeriesSplit(n_splits=N_INNER)
    search_final = GridSearchCV(
        pipe_full,
        param_grid=grid,
        cv=final_cv,
        scoring="r2",
        n_jobs=-1,
        refit=True
    )
    
    # Clip full data before final fit 
    X_final = _clip_frame(X)
    y_final = _clip_series(y)
    #------------------------------------------------------------------------------------------
    
    # Retrain the best model on the full dataset for final interpretation (not evaluation)
    search_final.fit(X_final, y_final)
    
    final_model = search_final.best_estimator_

    # Save best hyperparameters
    best_params = search_final.best_params_
    pd.DataFrame(list(best_params.items()), columns=["Parameter", "Value"]).to_csv(
        os.path.join(out_dir, f"best_model_hyperparams_{best_overall_model}.csv"), index=False
    )

    yhat_full = final_model.predict(X)
    plot_diagnostics(y, yhat_full, best_overall_model + "_FULL", "FullData")

    # Permutation importance
    result = permutation_importance(final_model, X, y, n_repeats=20, random_state=RSEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": X.columns,
                        "importance": result.importances_mean,
                        "importance_std": result.importances_std}).sort_values("importance", ascending=False)
    imp.to_csv(os.path.join(out_dir, f"final_perm_importance_{best_overall_model}.csv"), index=False)

    plt.figure(figsize=(8,10))
    sns.barplot(data=imp.head(20), x="importance", y="feature", orient="h")
    plt.title(f"{region_name.upper()}: Permutation Importance (Top 20) — {best_overall_model}")
    plt.xlabel("Mean importance (Δ score)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"final_perm_importance_{best_overall_model}.png"), dpi=300)
    plt.close()
    #------------------------------------------------------------------------------------------
    
    # Baseline Linear — outer-CV summary + CSV
    pipe_baseline = Pipeline([("prep", preprocess), ("model", LinearRegression())])
    r2s, rmses, maes = [], [], []
    for tr, te in outer_cv.split(X):
        Xtr_b, Xte_b = _clip_frame(X.iloc[tr]), _clip_frame(X.iloc[te])
        ytr_b, yte_b = _clip_series(y.iloc[tr]), _clip_series(y.iloc[te])
    
        pipe_baseline.fit(Xtr_b, ytr_b)
        yhat = pipe_baseline.predict(Xte_b)
    
        r2s.append(r2_score(yte_b, yhat))
        rmses.append(mean_squared_error(yte_b, yhat, squared=False))
        maes.append(mean_absolute_error(yte_b, yhat))
        
    baseline_summary = pd.DataFrame({"R2": r2s, "RMSE": rmses, "MAE": maes})
    baseline_summary.loc["Mean"] = baseline_summary.mean()
    baseline_summary.to_csv(os.path.join(out_dir, "baseline_linear_results.csv"), index=False)
    #------------------------------------------------------------------------------------------
    
    # Save final model and metadata
    try:
        import joblib, json

        # Save model (.pkl)
        model_path = os.path.join(out_dir, f"final_model_{region_name}.pkl")
        joblib.dump(final_model, model_path)

        # Prepare metadata summary
        metadata = {
            "Region": region_name,
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Num_Features": X.shape[1],
            "Best_Model": best_overall_model,
            "Best_Hyperparameters": best_params,
            "CrossVal_Metrics": {
                "Mean_R2": float(by_model_all.loc[best_overall_model, "R2"]),
                "Mean_RMSE": float(by_model_all.loc[best_overall_model, "RMSE"]),
                "Mean_MAE": float(by_model_all.loc[best_overall_model, "MAE"])
            },
            "Files_Generated": {
                "Model_Pickle": os.path.basename(model_path),
                "Hyperparams_CSV": f"best_model_hyperparams_{best_overall_model}.csv",
                "Permutation_Importance_CSV": f"final_perm_importance_{best_overall_model}.csv"
            }
        }

        # Save metadata JSON
        meta_path = os.path.join(out_dir, f"final_model_metadata_{region_name}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[{region_name}] Final model and metadata saved.")
    except Exception as e:
        print(f"[{region_name}] Model saving failed: {e}")
    #------------------------------------------------------------------------------------------
    
    # PDPs for Top-3 features of final model
    try:
        top3 = imp.head(3)["feature"].tolist()
        fig, ax = plt.subplots(figsize=(12, 4))
        disp = PartialDependenceDisplay.from_estimator(
            final_model, X, features=top3, kind="average", ax=ax, n_jobs=-1
        )
        plt.suptitle(f"{region_name.upper()}: PDP — Top 3 ({best_overall_model})", y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pdp_top3_{best_overall_model}.png"), dpi=300)
        plt.close()

        # Save PDP data 
        try:
        
            # Unpack the single element array
            lines_group = disp.lines_[0]  # shape (3,), each is a line2D
            for j, line in enumerate(lines_group):
                if hasattr(line, "get_xdata"):
                    values = line.get_xdata()
                    averages = line.get_ydata()
                    pd.DataFrame({
                        "region": region_name,
                        "feature": top3[j],
                        "feature_value": values,
                        "partial_dependence": averages
                    }).to_csv(
                        os.path.join(out_dir, f"pdp_data_{top3[j]}_{best_overall_model}.csv"),
                        index=False
                    )

            print(f"[{region_name}] PDP data saved successfully for: {top3}")
        
        except Exception as e:
            print(f"[{region_name}] PDP data export failed inside PDP block: {e}")
        

    except Exception as e:
        print(f"[{region_name}] PDP export failed: {e}")
    #------------------------------------------------------------------------------------------
    
    # Baseline vs Best — comparison bars
    # Use by_model_all (averaged across all folds), not by_model (fold winners only)
    comparison_df = pd.DataFrame({
        "Model": ["Baseline_Linear", best_overall_model],
        "Mean_R2": [baseline_summary.loc["Mean", "R2"], by_model_all.loc[best_overall_model, "R2"]],
        "Mean_RMSE": [baseline_summary.loc["Mean", "RMSE"], by_model_all.loc[best_overall_model, "RMSE"]],
        "Mean_MAE": [baseline_summary.loc["Mean", "MAE"], by_model_all.loc[best_overall_model, "MAE"]],
    })

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); sns.barplot(data=comparison_df, x="Model", y="Mean_R2", palette="Blues_d"); plt.title("Mean R²")
    plt.subplot(1, 3, 2); sns.barplot(data=comparison_df, x="Model", y="Mean_RMSE", palette="Greens_d"); plt.title("Mean RMSE")
    plt.subplot(1, 3, 3); sns.barplot(data=comparison_df, x="Model", y="Mean_MAE", palette="Oranges_d"); plt.title("Mean MAE")
    plt.tight_layout()
    comparison_df.to_csv(os.path.join(out_dir, "baseline_vs_best_metrics.csv"), index=False)

    plt.savefig(os.path.join(out_dir, "baseline_vs_best_metrics.png"), dpi=300)
    plt.close()
    #------------------------------------------------------------------------------------------
    
    # Baseline vs Best — multi-metric improvement
    base_r2 = baseline_summary.loc["Mean", "R2"]
    base_rmse = baseline_summary.loc["Mean", "RMSE"]
    base_mae = baseline_summary.loc["Mean", "MAE"]
    
    # Lookup for best model metrics (using overall averages)
    if "model" in by_model_all.columns:
        row_best = by_model_all.loc[by_model_all["model"] == best_overall_model]
        if not row_best.empty:
            best_r2 = row_best["R2"].values[0]
            best_rmse = row_best["RMSE"].values[0]
            best_mae = row_best["MAE"].values[0]
        else:
            print(f"[{region_name}] Best model '{best_overall_model}' not found in by_model_all.")
            best_r2 = best_rmse = best_mae = np.nan
    else:
        if best_overall_model in by_model_all.index:
            best_r2 = by_model_all.loc[best_overall_model, "R2"]
            best_rmse = by_model_all.loc[best_overall_model, "RMSE"]
            best_mae = by_model_all.loc[best_overall_model, "MAE"]
        else:
            print(f"[{region_name}] Best model '{best_overall_model}' missing from index.")
            best_r2 = best_rmse = best_mae = np.nan
    
    # Compute improvements 
    if not np.isnan(best_r2):
        delta_r2 = best_r2 - base_r2
        rmse_reduction_pct = ((base_rmse - best_rmse) / base_rmse) * 100
        mae_reduction_pct = ((base_mae - best_mae) / base_mae) * 100
    else:
        delta_r2 = rmse_reduction_pct = mae_reduction_pct = np.nan
    
    improvement_summary = pd.DataFrame({
        "Region": [region_name],
        "Baseline_R2": [base_r2],
        "Best_Model": [best_overall_model],
        "Best_Model_R2": [best_r2],
        "Delta_R2": [delta_r2],
        "Baseline_RMSE": [base_rmse],
        "Best_Model_RMSE": [best_rmse],
        "RMSE_Reduction_%": [rmse_reduction_pct],
        "Baseline_MAE": [base_mae],
        "Best_Model_MAE": [best_mae],
        "MAE_Reduction_%": [mae_reduction_pct]
    })
    improvement_csv = os.path.join(out_dir, f"baseline_comparison_{best_overall_model}.csv")
    improvement_summary.to_csv(improvement_csv, index=False)

    return dict(region=region_name, best_model=best_overall_model,
                all_models_summary=by_model_all, best_model_summary=by_model)
#------------------------------------------------------------------------------------------

# Run all regions and summary
def main():
    summaries = []
    for cfg in REGION_CONFIG:
        print(f"\n==== Running region: {cfg['name'].upper()} ====")
        res = run_region(cfg["name"], cfg["csv"], cfg["prefix"])
        summaries.append(res)

    # Winners table and plot
    rows = []
    for r in summaries:
        # Use all_models_summary, not best_model_summary
        df = r["all_models_summary"]
        best_row = df.loc[df["R2"].idxmax()]
        rows.append({
            "Region": r["region"],
            "Winner": best_row.name,  
            "Mean_R2": best_row["R2"],
            "Mean_RMSE": best_row["RMSE"],
            "Mean_MAE": best_row["MAE"]
        })

    cross = pd.DataFrame(rows).sort_values("Region")
    cross.to_csv(os.path.join(OUT_ROOT, "cross_region_winners.csv"), index=False)
    print("\nCross-Region Winners")
    print(cross)    
    #------------------------------------------------------------------------------------------
    
    # Consistency Check
    try:
        comp_all = pd.read_csv(os.path.join(OUT_ROOT, "cross_region_r2_comparison.csv"))
        check = (
            comp_all.groupby("Region", as_index=False)
            .apply(lambda g: g.loc[g["Mean_R2"].idxmax(), ["model", "Mean_R2"]])
            .reset_index(drop=True)
        )
        print("\n[Consistency Check] Top model per region from cross_region_r2_comparison.csv:")
        print(check)
    except Exception as e:
        print(f"[Consistency Check skipped] Reason: {e}")
    
    plt.figure(figsize=(7,4))
    sns.barplot(data=cross, x="Region", y="Mean_R2", hue="Winner", dodge=False)
    plt.title("Best Model per Region (Mean R²)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_ROOT, "cross_region_winners.png"), dpi=300)
    plt.close()

    #------------------------------------------------------------------------------------------
    # Cross-Region R2 comparison across all Models
    comp_csv = os.path.join(OUT_ROOT, "cross_region_r2_comparison.csv")
    if os.path.exists(comp_csv):
        os.remove(comp_csv)  #Avoid mixing with stale data

    all_model_dfs = []
    missing = []
    for cfg in REGION_CONFIG:
        region_dir = os.path.join(OUT_ROOT, cfg["name"])
        all_path = os.path.join(region_dir, "all_models_summary.csv")

        if not os.path.exists(all_path):
            missing.append(cfg["name"])
            continue

        # Re-read the file just written in this run
        tmp = pd.read_csv(all_path)
        if "model" not in tmp.columns:
            tmp = tmp.reset_index().rename(columns={"index": "model"})

        # Enure expected columns exist and types are numeric
        if "R2" not in tmp.columns:
            raise ValueError(f"'R2' not found in {all_path}. Columns: {list(tmp.columns)}")

        # Ensure numeric dtype 
        tmp["R2"] = pd.to_numeric(tmp["R2"], errors="coerce")
        tmp["RMSE"] = pd.to_numeric(tmp["RMSE"], errors="coerce")
        tmp["MAE"] = pd.to_numeric(tmp["MAE"], errors="coerce")

        tmp["Region"] = cfg["name"]
        all_model_dfs.append(tmp)

    if missing:
        print(f"Skipping regions with no summary: {missing}")

    if all_model_dfs:
        df_all = pd.concat(all_model_dfs, ignore_index=True)

        df_all.rename(columns={"R2": "Mean_R2", "RMSE": "Mean_RMSE", "MAE": "Mean_MAE"}, inplace=True)
        df_all.to_csv(comp_csv, index=False)

        # Plot
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_all, x="Region", y="Mean_R2", hue="model", palette="deep", edgecolor="black")
        plt.title("Cross-Region R² Comparison Across All Models")
        plt.ylabel("Mean R² (Outer CV)")
        plt.xlabel("Region")
        plt.legend(title="model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_ROOT, "cross_region_r2_comparison.png"), dpi=300)        
    #------------------------------------------------------------------------------------------
    
    # Cross-Region baseline vs best Comparison (using % MAE Reduction)
    comp_improvement_rows = []
    for cfg in REGION_CONFIG:
        region_dir = os.path.join(OUT_ROOT, cfg["name"])
        csvs = [f for f in os.listdir(region_dir)
                if f.startswith("baseline_comparison_") and f.endswith(".csv")]
        if not csvs:
            print(f"[{cfg['name']}] No baseline comparison file found.")
            continue
        comp_path = os.path.join(region_dir, csvs[0])
        df_imp = pd.read_csv(comp_path)
        comp_improvement_rows.append(df_imp)

    if comp_improvement_rows:
        comp_improvement = pd.concat(comp_improvement_rows, ignore_index=True)
        comp_csv = os.path.join(OUT_ROOT, "cross_region_baseline_comparison.csv")
        comp_improvement.to_csv(comp_csv, index=False)

        # Plot % MAE Reduction across regions
        plt.figure(figsize=(7, 4))
        sns.barplot(
            data=comp_improvement,
            x="Region",
            y="MAE_Reduction_%",
            hue="Best_Model",
            dodge=False,
            palette="viridis",
            edgecolor="black"
        )
        plt.title("Cross-Region % MAE Reduction (Best vs Baseline)")
        plt.ylabel("% Reduction in MAE")
        plt.xlabel("Region")
        plt.legend(title="Best Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_ROOT, "cross_region_mae_reduction.png"), dpi=300)
        plt.close()

        print("\nCross-Region Baseline vs Best Summary (using MAE Reduction)")
        print(comp_improvement[["Region", "Best_Model", "Delta_R2", "RMSE_Reduction_%", "MAE_Reduction_%"]])
    else:
        print("No baseline_comparison_*.csv files found — skipping cross-region improvement summary.")

if __name__ == "__main__":
    main()

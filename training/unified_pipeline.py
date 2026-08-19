#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========================================================================
                  Unified Pharmacokinetic Modeling Pipeline
         Trains and Validates on BOTH Kaggle CSV & Ross NONMEM datasets
===========================================================================
"""
# ===========================================================================
# PATH: unified_pipeline.py
# Unified Pharmacokinetic Modeling Pipeline: Trains & Validates ensembled ML models
# ===========================================================================

import os
import argparse
import time
import warnings
import joblib
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

T_MAX_REF = 240.0
LLOQ = 0.1
LINEAR_BASELINES = {"LinearRegression", "Ridge", "Lasso", "ElasticNet"}

def load_and_merge_data(file_paths):
    print("Loading and merging datasets...")
    dfs = []
    
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"  [WARN] File not found, skipping: {file_path}")
            continue
            
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        is_numeric_first_token = first_line and first_line.split()[0].replace('.', '', 1).replace('-', '', 1).isdigit()
        
        if is_numeric_first_token:
            print(f"  Parsing NONMEM dataset: {file_path}")
            default_cols = ['ID', 'TIME', 'AMT', 'RATE', 'DV', 'BQL', 'SEX', 'OCC', 'C9', 'HGHT', 'WGHT', 'C12', 'C13', 'Z1', 'Z2']
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=default_cols)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['DATASET'] = 'Ross'
            df['ID'] = df['ID'] + 10000
            if 'AGE' not in df.columns: df['AGE'] = 40.0
        else:
            print(f"  Parsing CSV dataset: {file_path}")
            df = pd.read_csv(file_path)
            df['DATASET'] = 'Kaggle'
            df['ID'] = df['Subject'] if 'Subject' in df.columns else df.get('ID', 1)
            rename_map = {'Time': 'TIME', 'Amt': 'AMT', 'Rate': 'RATE', 'conc': 'DV', 'Ht': 'HGHT', 'Wt': 'WGHT', 'Age': 'AGE', 'Sex': 'SEX'}
            df = df.rename(columns=rename_map)
            
        dfs.append(df)
        
    if not dfs:
        raise FileNotFoundError("No valid datasets found. Please check file paths.")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined raw data shape : {combined_df.shape[0]} rows × {combined_df.shape[1]} columns")
    return combined_df

def feature_engineering(df):
    print("Performing feature engineering on combined corpus...")

    drop_cols = ['rownames', 'Subject', 'LBM', 'BSA', 'C9', 'C12', 'C13', 'Z1', 'Z2', 'MDV', 'BQL', 'OCC', 'DATASET']
    df = df.drop(columns=drop_cols, errors='ignore')

    rename_map = {'Time': 'TIME', 'Amt': 'AMT', 'Rate': 'RATE', 'conc': 'DV', 'Ht': 'HGHT', 'Wt': 'WGHT', 'Age': 'AGE', 'Sex': 'SEX'}
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]

    df.loc[(df['AMT'] > 0) & (df['TIME'] == 0) & (df['DV'].isna()), 'DV'] = np.nan
    df.loc[(df['AMT'] > 0) & (df['DV'] == 0), 'DV'] = np.nan

    df_model = df.dropna(subset=['DV']).copy()
    df_model = df_model[df_model['DV'] <= 25].copy()
    df_model.reset_index(drop=True, inplace=True)

    df_model_sorted = df_model.sort_values(['ID', 'TIME'])
    df_model['Cum_Dose'] = df_model_sorted.groupby('ID')['AMT'].cumsum().reindex(df_model.index).fillna(0)

    df_model['is_bql'] = ((df_model['Cum_Dose'] == 0) & (df_model['RATE'] == 0)).astype(int)
    _bql_mask = df_model['DV'] <= LLOQ
    df_model.loc[_bql_mask, 'DV'] = LLOQ / 2.0
    df_model.loc[_bql_mask, 'is_bql'] = 1

    if 'SEX' not in df_model.columns:
        df_model['SEX'] = 1
    else:
        df_model['SEX'] = pd.to_numeric(df_model['SEX'], errors='coerce').fillna(1).astype(int)

    if 'AGE' not in df_model.columns: df_model['AGE'] = 40.0

    for col in ['WGHT', 'HGHT']:
        if col in df_model.columns and (df_model[col] <= 0).any():
            df_model.loc[df_model[col] <= 0, col] = np.nan

    df_model['BMI'] = df_model['WGHT'] / ((df_model['HGHT'] / 100) ** 2)
    lbm_denom = np.where(df_model['SEX'] == 1, 6680 + 216 * df_model['BMI'], 8780 + 244 * df_model['BMI'])
    df_model['LBM'] = np.where((df_model['WGHT'] > 0) & (lbm_denom > 0), 9270 * df_model['WGHT'] / lbm_denom, np.nan)
    df_model['BSA'] = np.where((df_model['WGHT'] > 0) & (df_model['HGHT'] > 0), np.sqrt(df_model['HGHT'] * df_model['WGHT'] / 3600.0), np.nan)

    _eps = 1e-6
    df_model['Dose_per_Kg'] = np.where(df_model['WGHT'] > 0, df_model['AMT'] / df_model['WGHT'], np.nan)
    df_model['Rate_per_Kg'] = np.where(df_model['WGHT'] > 0, df_model['RATE'] / df_model['WGHT'], np.nan)
    df_model['Dose_per_LBM'] = np.where(df_model['LBM'] > 0, df_model['AMT'] / df_model['LBM'], np.nan)
    df_model['Rate_per_LBM'] = np.where(df_model['LBM'] > 0, df_model['RATE'] / df_model['LBM'], np.nan)
    df_model['Dose_per_BSA'] = np.where(df_model['BSA'] > 0, df_model['AMT'] / df_model['BSA'], np.nan)
    df_model['logTIME'] = np.log1p(df_model['TIME'])
    df_model['TIME_SQ'] = df_model['TIME'] ** 2
    df_model['Dose_BMI'] = df_model['AMT'] * df_model['BMI']
    df_model['log_RATE'] = np.log1p(df_model['RATE'].clip(lower=0))
    df_model['log_AMT']  = np.log1p(df_model['AMT'].clip(lower=0))
    df_model['TIME_RATE'] = df_model['TIME'] * df_model['RATE']
    df_model['TIME_AMT']  = df_model['TIME'] * df_model['AMT']
    df_model['Rate_per_Dose']   = df_model['RATE'] / (df_model['AMT'] + _eps)
    df_model['Dose_Rate_ratio'] = df_model['AMT']  / (df_model['RATE'] + _eps)
    df_model['TIME_CBRT'] = np.cbrt(df_model['TIME'])
    df_model['invTIME']   = 1.0 / (df_model['TIME'] + _eps)
    df_model['TIME_norm'] = df_model['TIME'] / (T_MAX_REF + _eps)
    _total_amt = df_model.groupby('ID')['AMT'].transform('sum').replace(0, _eps)
    df_model['Dose_frac'] = df_model['Cum_Dose'] / _total_amt

    _sex_factor = np.where(df_model['SEX'] == 1, 1.0, 0.85)
    df_model['eGFR_proxy'] = ((140 - df_model['AGE']) * df_model['WGHT'] / 72.0 * _sex_factor).clip(lower=0)
    df_model['Hepatic_proxy'] = df_model['WGHT'] / (df_model['AGE'] * df_model['BMI'] + _eps)

    df_model['is_infusion'] = (df_model['RATE'] > 0).astype(int)
    _inf_end = (df_model[df_model['RATE'] > 0].groupby('ID')['TIME'].max().rename('_inf_end_time'))
    df_model = df_model.join(_inf_end, on='ID')
    df_model['_inf_end_time'] = df_model['_inf_end_time'].fillna(0)
    df_model['time_post_inf'] = (df_model['TIME'] - df_model['_inf_end_time']).clip(lower=0)
    df_model['phase_ratio']   = df_model['time_post_inf'] / (df_model['TIME'] + _eps)
    df_model = df_model.drop(columns=['_inf_end_time'])

    X = df_model.drop(columns=['DV'])
    groups = X['ID']
    X = X.drop(columns=['ID'])
    y = df_model['DV']

    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    print(f"  Engineered feature matrix : {X.shape[0]} rows × {X.shape[1]} features")
    return X, y, groups

def create_models():
    def _qt(estimator):
        return TransformedTargetRegressor(
            regressor=estimator,
            transformer=QuantileTransformer(output_distribution='normal', random_state=RANDOM_SEED)
        )

    xgb_tuned = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.0, reg_lambda=0.1, min_child_weight=5, gamma=0.1, tree_method='hist', random_state=RANDOM_SEED, verbosity=0, objective='reg:absoluteerror')
    lgb_tuned = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.6, min_child_samples=20, reg_lambda=0.1, reg_alpha=1.0, deterministic=True, force_col_wise=True, num_threads=1, verbose=-1, objective='mae')
    et_tuned = ExtraTreesRegressor(n_estimators=500, max_depth=20, max_features='sqrt', min_samples_split=5, min_samples_leaf=2, bootstrap=False, random_state=RANDOM_SEED, criterion='absolute_error')
    rf_tuned = RandomForestRegressor(n_estimators=300, max_depth=30, max_features='log2', min_samples_split=10, min_samples_leaf=2, bootstrap=True, random_state=RANDOM_SEED, criterion='absolute_error')
    gbm_tuned = GradientBoostingRegressor(loss='absolute_error', n_estimators=1000, learning_rate=0.02, max_depth=3, subsample=0.85, min_samples_leaf=15, max_features='sqrt', random_state=RANDOM_SEED)

    models = {
        "BayesianCompartment": _qt(BayesianRidge()),
        "SupportVectorRobust": _qt(SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1)),
        "RandomForest":     _qt(rf_tuned),
        "ExtraTrees":       _qt(et_tuned),
        "GradientBoosting": _qt(gbm_tuned),
        "XGBoost":          _qt(xgb_tuned),
        "LightGBM":         _qt(lgb_tuned),
        "CatBoost":         _qt(cb.CatBoostRegressor(iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=9, subsample=0.7, random_state=RANDOM_SEED, verbose=0)),
        "LinearRegression": _qt(LinearRegression()),
        "Ridge":            _qt(Ridge()),
        "Lasso":            _qt(Lasso()),
        "ElasticNet":       _qt(ElasticNet()),
    }

    from sklearn.base import clone
    stacking_estimators = [('xgb', clone(xgb_tuned)), ('et', clone(et_tuned)), ('rf', clone(rf_tuned)), ('gbm', clone(gbm_tuned))]
    models["StackingEnsemble"] = _qt(StackingRegressor(estimators=stacking_estimators, final_estimator=HuberRegressor(epsilon=1.35, alpha=1.0), passthrough=False, cv=5))
    models["StackingPassthrough"] = _qt(StackingRegressor(estimators=stacking_estimators, final_estimator=HuberRegressor(epsilon=1.35, alpha=1.0), passthrough=True, cv=5))
    return models

def main(data_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_and_merge_data(data_paths)
    X, y, groups = feature_engineering(df)
    
    print("\nSplitting data (GroupShuffleSplit)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = create_models()
    results = []
    
    print("\nTraining models...")
    for name, model in models.items():
        try:
            t0 = time.time()
            model.fit(X_train_scaled, y_train)
            t1 = time.time()
            
            y_pred = np.clip(model.predict(X_test_scaled), 0, y_test.max() * 1.5)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            print(f"{name:>20}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
            results.append({"Model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "Train_Time_s": round(t1-t0, 4)})
        except Exception as e:
            print(f"  [SKIP] {name} failed: {e}")

    print("\nSaving artifacts...")
    ARTIFACT_DIR = os.path.join(output_dir, "models")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    joblib.dump(list(X.columns), os.path.join(ARTIFACT_DIR, "feature_names.joblib"))
    for name, model in models.items():
        try:
            joblib.dump(model, os.path.join(ARTIFACT_DIR, f"{name}.joblib"))
        except Exception as e:
            print(f"  [SKIP] Could not save {name}: {e}")

    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
    results_df.to_csv(os.path.join(output_dir, "unified_pipeline_results.csv"), index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified PK Modeling Pipeline")
    parser.add_argument("--data", nargs="+", default=["Kaggle_PK_Remifentanil.csv", "Ross_remi.txt"], help="Paths to datasets")
    parser.add_argument("--outdir", type=str, default="./kaggle_remi_output", help="Output directory")
    args = parser.parse_args()
    main(args.data, args.outdir)

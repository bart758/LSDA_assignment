import argparse

import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from common import PREPROCESSING, build_pipelines, load_data, extract_model_params, set_tracking_uri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power_data", type=str, default="data/power.csv")
    parser.add_argument("--weather_data", type=str, default="data/weather.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    set_tracking_uri()
    mlflow.set_experiment("CV-Strategy-Comparison")

    joined_dfs = load_data(args.power_data, args.weather_data)

    _, pipeline_rf, _ = build_pipelines()
    rf_params = extract_model_params(pipeline_rf)

    cv_strategies = {
        "TimeSeriesSplit": TimeSeriesSplit(n_splits=args.n_splits),
        "RandomKFold":      KFold(n_splits=args.n_splits, shuffle=True, random_state=42),
    }

    with mlflow.start_run(run_name="RF_cv_strategy_comparison"):
        strategy_maes = {}

        for strategy_name, cv in cv_strategies.items():
            fold_maes = []
            for train_idx, val_idx in cv.split(joined_dfs):
                X_tr = joined_dfs.iloc[train_idx]
                y_tr = joined_dfs["Total"].iloc[train_idx]
                X_va = joined_dfs.iloc[val_idx]
                y_va = joined_dfs["Total"].iloc[val_idx]

                pipe = Pipeline(PREPROCESSING + [
                    ("model", RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1))
                ])
                pipe.fit(X_tr, y_tr)
                fold_maes.append(mean_absolute_error(pipe.predict(X_va), y_va))

            cv_mae = float(np.mean(fold_maes))
            strategy_maes[strategy_name] = cv_mae

            with mlflow.start_run(run_name=f"RF_{strategy_name}", nested=True):
                mlflow.log_param("cv_strategy", strategy_name)
                mlflow.log_params(rf_params)
                mlflow.log_metric("cv_mae", cv_mae)

        leakage_gap = strategy_maes["RandomKFold"] - strategy_maes["TimeSeriesSplit"]

        mlflow.log_metric("timeseries_cv_mae", strategy_maes["TimeSeriesSplit"])
        mlflow.log_metric("random_kfold_cv_mae", strategy_maes["RandomKFold"])
        mlflow.log_metric("leakage_gap", leakage_gap)
        mlflow.set_tag("summary", "true")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(strategy_maes.keys(), strategy_maes.values(), color=["steelblue", "indianred"])
        ax.set_ylabel("CV MAE (MW)")
        ax.set_title("CV strategy comparison -- leakage check")
        plt.tight_layout()
        mlflow.log_figure(fig, "cv_strategy_comparison.png")
        plt.close(fig)

    print(f"TimeSeriesSplit MAE: {strategy_maes['TimeSeriesSplit']:.4f} MW")
    print(f"Random KFold MAE:    {strategy_maes['RandomKFold']:.4f} MW")
    print(f"Gap (negative = leakage signal): {leakage_gap:.4f} MW")


if __name__ == "__main__":
    main()

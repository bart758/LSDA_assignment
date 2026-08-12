import argparse

import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

from common import PREPROCESSING, build_pipelines, load_data, extract_model_params, set_tracking_uri


def cv_mae_for_seed(model_class, params, fixed_kwargs, seed, joined_dfs, tscv):
    fold_maes = []
    for train_idx, val_idx in tscv.split(joined_dfs):
        X_tr = joined_dfs.iloc[train_idx]
        y_tr = joined_dfs["Total"].iloc[train_idx]
        X_va = joined_dfs.iloc[val_idx]
        y_va = joined_dfs["Total"].iloc[val_idx]

        pipe = Pipeline(PREPROCESSING + [
            ("model", model_class(**params, **fixed_kwargs, random_state=seed))
        ])
        pipe.fit(X_tr, y_tr)
        fold_maes.append(mean_absolute_error(pipe.predict(X_va), y_va))
    return float(np.mean(fold_maes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power_data", type=str, default="data/power.csv")
    parser.add_argument("--weather_data", type=str, default="data/weather.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="0,1,42,123,2024")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    set_tracking_uri()
    mlflow.set_experiment("Seed-Variance-Check")

    joined_dfs = load_data(args.power_data, args.weather_data)
    tscv = TimeSeriesSplit(n_splits=args.n_splits)

    # Uses the project's default hyperparameters (same ones as train.py's pipelines)
    _, pipeline_rf, pipeline_nn = build_pipelines()
    rf_params = extract_model_params(pipeline_rf)
    mlp_params = extract_model_params(pipeline_nn)

    model_configs = [
        ("RandomForest", RandomForestRegressor, rf_params, {"n_jobs": -1}),
        ("MLP",          MLPRegressor,          mlp_params, {}),
    ]

    for model_name, model_class, params, fixed_kwargs in model_configs:
        with mlflow.start_run(run_name=f"{model_name}_seed_variance"):
            seed_maes = []
            for seed in seeds:
                cv_mae = cv_mae_for_seed(model_class, params, fixed_kwargs, seed, joined_dfs, tscv)
                seed_maes.append(cv_mae)

                with mlflow.start_run(run_name=f"{model_name}_seed{seed}", nested=True):
                    mlflow.log_params({**params, **fixed_kwargs, "random_state": seed})
                    mlflow.log_metric("cv_mae", cv_mae)

            mean_mae = float(np.mean(seed_maes))
            std_mae = float(np.std(seed_maes))

            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_mae", mean_mae)
            mlflow.log_metric("std_cv_mae", std_mae)
            mlflow.set_tag("summary", "true")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar([str(s) for s in seeds], seed_maes, color="steelblue")
            ax.axhline(mean_mae, color="red", linestyle="--", label=f"mean={mean_mae:.3f}")
            ax.set_xlabel("random_state seed")
            ax.set_ylabel("CV MAE (MW)")
            ax.set_title(f"{model_name}: MAE stability across seeds")
            ax.legend()
            plt.tight_layout()
            mlflow.log_figure(fig, f"{model_name}_seed_variance.png")
            plt.close(fig)

        print(f"{model_name}: mean_cv_mae={mean_mae:.4f} +/- {std_mae:.4f} MW")


if __name__ == "__main__":
    main()

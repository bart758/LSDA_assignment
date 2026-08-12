import argparse
from itertools import product

import numpy as np
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from common import PREPROCESSING, load_data, set_tracking_uri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power_data", type=str, default="data/power.csv")
    parser.add_argument("--weather_data", type=str, default="data/weather.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    set_tracking_uri()
    mlflow.set_experiment("RandomForestRegressor-GridSearch")

    joined_dfs = load_data(args.power_data, args.weather_data)
    tscv = TimeSeriesSplit(n_splits=args.n_splits)

    param_grid = {
        "n_estimators":     [100, 300, 500],
        "max_depth":        [5, 8, None],
        "min_samples_leaf": [1, 5, 10],
        "max_features":     ["sqrt", 1.0],
    }

    best_mae = float("inf")
    best_params = {}

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    print(f"Running {len(combos)} combinations x {tscv.n_splits} folds ...\n")

    with mlflow.start_run(run_name="RF_gridsearch"):
        for combo in combos:
            params = dict(zip(keys, combo))
            fold_maes = []

            for train_idx, val_idx in tscv.split(joined_dfs):
                X_tr = joined_dfs.iloc[train_idx]
                y_tr = joined_dfs["Total"].iloc[train_idx]
                X_va = joined_dfs.iloc[val_idx]
                y_va = joined_dfs["Total"].iloc[val_idx]

                pipe = Pipeline(PREPROCESSING + [
                    ("model", RandomForestRegressor(**params, random_state=42, n_jobs=-1))
                ])
                pipe.fit(X_tr, y_tr)
                fold_maes.append(mean_absolute_error(pipe.predict(X_va), y_va))

            cv_mae = float(np.mean(fold_maes))
            run_name = (
                f"RF_n={params['n_estimators']}"
                f"_depth={params['max_depth']}"
                f"_leaf={params['min_samples_leaf']}"
                f"_feat={params['max_features']}"
            )

            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_mae", cv_mae)
                mlflow.log_param("n_splits", tscv.n_splits)

            print(f"  {run_name:80s}  cv_mae={cv_mae:.4f} MW")

            if cv_mae < best_mae:
                best_mae = cv_mae
                best_params = params

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_mae", best_mae)
        mlflow.set_tag("summary", "true")

    print(f"\nBest cv_mae: {best_mae:.4f} MW")
    print(f"Best params: {best_params}")


if __name__ == "__main__":
    main()

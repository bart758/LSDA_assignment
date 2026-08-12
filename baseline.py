import argparse

import numpy as np
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import load_data, split_train_test, set_tracking_uri

# (experiment_name, run_name, registered_model_name) for each model trained by train.py
MODEL_RUNS = [
    ("WindPower-LinearRegression", "LR_run", "WindPower-LR"),
    ("WindPower-RandomForest",     "RF_run", "WindPower-RF"),
    ("WindPower-NeuralNetwork",    "NN_run", "WindPower-NN"),
]


def get_latest_mae(client, experiment_name, run_name):
    """Looks up the most recent logged MAE for a given experiment/run name pair."""
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    return runs[0].data.metrics.get("MAE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power_data", type=str, default="data/power.csv")
    parser.add_argument("--weather_data", type=str, default="data/weather.csv")
    args = parser.parse_args()

    set_tracking_uri()
    client = MlflowClient()

    joined_dfs = load_data(args.power_data, args.weather_data)
    X_train, X_test, y_train, y_test = split_train_test(joined_dfs)

    mlflow.set_experiment("Baseline-Comparison")
    with mlflow.start_run(run_name="naive_baselines"):
        # Mean baseline: predict the training mean for every test point
        mean_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
        mean_mae = mean_absolute_error(y_test, mean_pred)
        mean_rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
        mean_r2 = r2_score(y_test, mean_pred)

        # Persistence baseline: predict the previous timestep's value
        persistence_pred = y_test.shift(1)
        persistence_pred.iloc[0] = y_train.iloc[-1]
        pers_mae = mean_absolute_error(y_test, persistence_pred)
        pers_rmse = np.sqrt(mean_squared_error(y_test, persistence_pred))
        pers_r2 = r2_score(y_test, persistence_pred)

        mlflow.log_param("baseline_types", "mean,persistence")
        mlflow.log_metric("mean_baseline_MAE", mean_mae)
        mlflow.log_metric("mean_baseline_RMSE", mean_rmse)
        mlflow.log_metric("mean_baseline_R2", mean_r2)
        mlflow.log_metric("persistence_baseline_MAE", pers_mae)
        mlflow.log_metric("persistence_baseline_RMSE", pers_rmse)
        mlflow.log_metric("persistence_baseline_R2", pers_r2)

        # Pull each trained model's test MAE from its own experiment/run.
        # Requires the `train` entry point to have been run at least once first.
        test_maes = {}
        for experiment_name, run_name, model_name in MODEL_RUNS:
            mae = get_latest_mae(client, experiment_name, run_name)
            if mae is not None:
                test_maes[model_name] = mae
            else:
                print(
                    f"Warning: no logged MAE found for {model_name} "
                    f"(run the 'train' entry point first) -- skipping from plot."
                )

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = ["Mean baseline", "Persistence baseline"] + list(test_maes.keys())
        values = [mean_mae, pers_mae] + list(test_maes.values())
        colors = ["gray", "gray"] + ["steelblue"] * len(test_maes)
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("MAE (MW)")
        ax.set_title("Model MAE vs. naive baselines")
        plt.xticks(rotation=15)
        plt.tight_layout()
        mlflow.log_figure(fig, "baseline_comparison.png")
        plt.close(fig)

    print(f"Mean baseline MAE:        {mean_mae:.4f} MW")
    print(f"Persistence baseline MAE: {pers_mae:.4f} MW")


if __name__ == "__main__":
    main()

import argparse

import numpy as np
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from common import build_pipelines, load_data, split_train_test, set_tracking_uri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power_data", type=str, default="data/power.csv")
    parser.add_argument("--weather_data", type=str, default="data/weather.csv")
    args = parser.parse_args()

    set_tracking_uri()
    mlflow.sklearn.autolog()

    joined_dfs = load_data(args.power_data, args.weather_data)
    X_train, X_test, y_train, y_test = split_train_test(joined_dfs)
    X_train_minimal = X_train[['time', 'Speed', 'Direction']]

    pipeline_lr, pipeline_rf, pipeline_nn = build_pipelines()

    experiments = [
        ("WindPower-LinearRegression", "LR_run", "WindPower-LR", pipeline_lr),
        ("WindPower-RandomForest",     "RF_run", "WindPower-RF", pipeline_rf),
        ("WindPower-NeuralNetwork",    "NN_run", "WindPower-NN", pipeline_nn),
    ]

    for experiment_name, run_name, model_name, pipeline in experiments:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", np.sqrt(mean_squared_error(y_test, preds)))
            mlflow.log_metric("R2", r2_score(y_test, preds))

            signature = infer_signature(X_train_minimal, preds)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                registered_model_name=model_name,
            )
            print(f"{model_name}: MAE={mae:.4f} MW")


if __name__ == "__main__":
    main()

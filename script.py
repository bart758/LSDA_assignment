import mlflow
import os

# You will probably need these
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import skops.io as sio

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from itertools import product
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import mlflow
from sklearn.model_selection import KFold



# convert cardinal direction into corresponding degree (assuming North is at deg= 0) 

class WindDirectionEncoder():

    def __init__(self):
        self.deg_conversion_ = {
            'N' : 0,
            'NNE' : 22.5,
            'NE' : 45,
            'ENE' : 67.5,
            'E' : 90,
            'ESE' : 112.5,
            'SE' : 135,
            'SSE' : 157.5,
            'S' : 180,
            'SSW' : 202.5,
            'SW' : 225,
            'WSW' : 247.5,
            'W' : 270,
            'WNW' : 292.5,
            'NW' : 315,
            'NNW' : 337.5
        }

    # no fitting needed, just to align with sklearn
    def fit(self, X= None, y=None):
        return self

    # apply transformation
    def transform(self, X: np.ndarray | pd.DataFrame, y=None):
        
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        X[:, 1] = pd.Series(X[:, 1]).map(self.deg_conversion_).to_numpy()
        return X

deg_conversion = {
    'N' : 0,
    'NNE' : 22.5,
    'NE' : 45,
    'ENE' : 67.5,
    'E' : 90,
    'ESE' : 112.5,
    'SE' : 135,
    'SSE' : 157.5,
    'S' : 180,
    'SSW' : 202.5,
    'SW' : 225,
    'WSW' : 247.5,
    'W' : 270,
    'WNW' : 292.5,
    'NW' : 315,
    'NNW' : 337.5
}

### TODO -> CREATE YOUR OWN PIPELINE ###
# Create your pipeline with the desired transformers



class MLflowJSONSanitizer(BaseEstimator, TransformerMixin):
    """
    Normalises raw or JSON-decoded input into a time-indexed DataFrame.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        df['_from_wind_df'] = 'time' in df.columns

        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], utc=True)
            df = df.set_index('time')
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        return df


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Drops every column except Speed, Direction, and the helper flag.
    """
    _KEEP = ['Speed', 'Direction', '_from_wind_df']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = [c for c in self._KEEP if c in df.columns]
        return df[cols]


class WindVectorEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes Direction as two unit-vector components (speed_u, speed_v).
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if df['Direction'].dtype == object:
            deg = df['Direction'].map(deg_conversion)
        else:
            deg = df['Direction']

        rad = np.deg2rad(deg)
        df['speed_u'] = df['Speed'] * np.sin(rad)   
        df['speed_v'] = df['Speed'] * np.cos(rad)

        df = df.drop(columns=['Direction', '_from_wind_df'], errors='ignore')
        return df


# ── Shared preprocessing steps ───────────────────────────────────────────────
_preprocessing = [
    ("sanitizer",   MLflowJSONSanitizer()),  # parse time → DatetimeIndex, add helper flag
    ("selector",    FeatureSelector()),       # keep Speed, Direction, helper flag
    ("speed_vector", WindVectorEncoder()),     # Direction and speed → (speed_u, speed_v);
    ("imputer", SimpleImputer(strategy="median")), #handles missing values
    ("scaler",      StandardScaler()),        # zero-mean, unit-variance
]

pipeline_lr = Pipeline(_preprocessing + [
    ("model", LinearRegression()),
])

pipeline_rf = Pipeline(_preprocessing + [
    ("model", RandomForestRegressor(max_depth=5, n_jobs=-1)),
])

RandomForestRegressor(max_depth=5, n_jobs=-1)


pipeline_nn = Pipeline(_preprocessing + [
    ("model", MLPRegressor(hidden_layer_sizes=(64, 32,), activation="tanh", alpha=0.001,learning_rate_init=0.01, max_iter=600)),
])

power_df = pd.read_csv('data/power.csv', parse_dates=["time"])
wind_df = pd.read_csv('data/weather.csv', parse_dates=["time"])
power_df.drop(['ANM', 'Non-ANM'], axis= 1, inplace= True)


pd.DataFrame(WindDirectionEncoder().transform(wind_df))
wind_df['Direction_Degree'] = wind_df.apply(lambda row: deg_conversion[row['Direction']], axis= 1)
power_3h = power_df.set_index('time').resample('3h').mean().reset_index()
joined_dfs = pd.merge_asof(power_3h, wind_df, on='time', direction='backward', tolerance=pd.Timedelta('3h')).dropna()

# Split the data so we can test how well our model performs in unseen data
# X_train, X_test, y_train, y_test = train_test_split(X, y) # -> You might want to use another split method

# 80% train / 20% test — no shuffling, temporal order preserved
X = joined_dfs.drop(columns=['Total'])
y = joined_dfs['Total']

split_idx = int(len(joined_dfs) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


mlflow.sklearn.autolog()

mlflow.set_tracking_uri("http://127.0.0.1:5000") # We set the MLFlow UI to display in our local host.

experiments = [
    ("WindPower-LinearRegression", "LR_run", "WindPower-LR", pipeline_lr),
    ("WindPower-RandomForest",     "RF_run", "WindPower-RF", pipeline_rf),
    ("WindPower-NeuralNetwork",    "NN_run", "WindPower-NN", pipeline_nn),
]

test_maes = {}  

X_train_minimal = X_train[['time', 'Speed', 'Direction']]
for experiment_name, run_name, model_name, pipeline in experiments:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)  

        mlflow.log_metric("MAE",  mean_absolute_error(y_test, preds))
        mlflow.log_metric("RMSE", np.sqrt(mean_squared_error(y_test, preds)))
        mlflow.log_metric("R2",   r2_score(y_test, preds))

        test_maes[model_name] = mae

        signature = infer_signature(X_train_minimal, preds)
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name,
        )


mlflow.sklearn.autolog(disable=True)

mlflow.set_experiment("MLPRegressor-GridSearch")

# Parameter grid
param_grid = {
    "hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32)],
    "activation":         ["relu", "tanh"],
    "alpha":              [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.01],
}

tscv = TimeSeriesSplit(n_splits=5)

best_mae   = float("inf")
best_params = {}

keys   = list(param_grid.keys())
combos = list(product(*param_grid.values()))
print(f"Running {len(combos)} combinations × {tscv.n_splits} folds …\n")

with mlflow.start_run(run_name="MLP_gridsearch") as parent_run:
    for combo in combos:
        params = dict(zip(keys, combo))
        fold_maes = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(joined_dfs)):
            X_tr = joined_dfs.iloc[train_idx]
            y_tr = joined_dfs["Total"].iloc[train_idx]
            X_va = joined_dfs.iloc[val_idx]
            y_va = joined_dfs["Total"].iloc[val_idx]

            pipe = Pipeline(_preprocessing + [
                ("model", MLPRegressor(
                    **params,
                    max_iter=500,
                    early_stopping=True,
                    random_state=42,
                ))
            ])
            pipe.fit(X_tr, y_tr)
            fold_maes.append(mean_absolute_error(pipe.predict(X_va), y_va))

        cv_mae = float(np.mean(fold_maes))

        run_name = (
            f"MLP_layers={'x'.join(str(n) for n in params['hidden_layer_sizes'])}"
            f"_act={params['activation']}"
            f"_alpha={params['alpha']}"
            f"_lr={params['learning_rate_init']}"
        )

        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_mae", cv_mae)
            mlflow.log_param("n_splits", tscv.n_splits)

        print(f"  {run_name:80s}  cv_mae={cv_mae:.4f} MW")

        if cv_mae < best_mae:
            best_mae    = cv_mae
            best_params = params

    # logged on the PARENT run — summarizes the whole search
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_cv_mae", best_mae)
    mlflow.set_tag("summary", "true")

print(f"\nBest cv_mae: {best_mae:.4f} MW")
print(f"Best params: {best_params}")


mlflow.set_experiment("RandomForestRegressor-GridSearch")

# Parameter grid
param_grid = {
    "n_estimators":     [100, 300, 500],
    "max_depth":        [5, 8, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features":     ["sqrt", 1.0],
}

tscv = TimeSeriesSplit(n_splits=5)

best_mae    = float("inf")
best_params = {}

keys   = list(param_grid.keys())
combos = list(product(*param_grid.values()))
print(f"Running {len(combos)} combinations × {tscv.n_splits} folds …\n")

with mlflow.start_run(run_name="RF_gridsearch") as parent_run:
    for combo in combos:
        params = dict(zip(keys, combo))
        fold_maes = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(joined_dfs)):
            X_tr = joined_dfs.iloc[train_idx]
            y_tr = joined_dfs["Total"].iloc[train_idx]
            X_va = joined_dfs.iloc[val_idx]
            y_va = joined_dfs["Total"].iloc[val_idx]

            pipe = Pipeline(_preprocessing + [
                ("model", RandomForestRegressor(
                    **params,
                    random_state=42,
                    n_jobs=-1,
                ))
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
            best_mae    = cv_mae
            best_params = params

    # logged on the PARENT run — summarizes the whole search
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_cv_mae", best_mae)
    mlflow.set_tag("summary", "true")

print(f"\nBest cv_mae: {best_mae:.4f} MW")
print(f"Best params: {best_params}")


mlflow.set_experiment("Baseline-Comparison")
with mlflow.start_run(run_name="naive_baselines"):
    # Mean baseline: predict the training mean for every test point
    mean_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    mean_mae  = mean_absolute_error(y_test, mean_pred)
    mean_rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
    mean_r2   = r2_score(y_test, mean_pred)

    # Persistence baseline: predict the previous timestep's value
    persistence_pred = y_test.shift(1)
    persistence_pred.iloc[0] = y_train.iloc[-1]
    pers_mae  = mean_absolute_error(y_test, persistence_pred)
    pers_rmse = np.sqrt(mean_squared_error(y_test, persistence_pred))
    pers_r2   = r2_score(y_test, persistence_pred)

    mlflow.log_param("baseline_types", "mean,persistence")
    mlflow.log_metric("mean_baseline_MAE",  mean_mae)
    mlflow.log_metric("mean_baseline_RMSE", mean_rmse)
    mlflow.log_metric("mean_baseline_R2",   mean_r2)
    mlflow.log_metric("persistence_baseline_MAE",  pers_mae)
    mlflow.log_metric("persistence_baseline_RMSE", pers_rmse)
    mlflow.log_metric("persistence_baseline_R2",   pers_r2)

    # Artifact: bar chart comparing baselines against your trained models
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Mean baseline", "Persistence baseline"] + list(test_maes.keys())
    values = [mean_mae, pers_mae] + list(test_maes.values())
    ax.bar(labels, values, color=["gray", "gray"] + ["steelblue"] * len(test_maes))
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Model MAE vs. naive baselines")
    plt.xticks(rotation=15)
    plt.tight_layout()
    mlflow.log_figure(fig, "baseline_comparison.png")
    plt.close(fig)

print(f"Mean baseline MAE:        {mean_mae:.4f} MW")
print(f"Persistence baseline MAE: {pers_mae:.4f} MW")



mlflow.set_experiment("Seed-Variance-Check")

seeds = [0, 1, 42, 123, 2024]

# Pull hyperparameters directly from your already-defined pipelines,
# excluding random_state/n_jobs since those get set per-seed/fixed separately
def extract_model_params(pipeline, exclude=("random_state", "n_jobs")):
    model_params = pipeline.named_steps["model"].get_params()
    return {k: v for k, v in model_params.items() if k not in exclude}

rf_params  = extract_model_params(pipeline_rf)
mlp_params = extract_model_params(pipeline_nn)

model_configs = [
    ("RandomForest", RandomForestRegressor, rf_params,  {"n_jobs": -1}),
    ("MLP",          MLPRegressor,          mlp_params, {}),
]

def cv_mae_for_seed(model_class, params, fixed_kwargs, seed):
    fold_maes = []
    for train_idx, val_idx in tscv.split(joined_dfs):
        X_tr = joined_dfs.iloc[train_idx]
        y_tr = joined_dfs["Total"].iloc[train_idx]
        X_va = joined_dfs.iloc[val_idx]
        y_va = joined_dfs["Total"].iloc[val_idx]

        pipe = Pipeline(_preprocessing + [
            ("model", model_class(**params, **fixed_kwargs, random_state=seed))
        ])
        pipe.fit(X_tr, y_tr)
        fold_maes.append(mean_absolute_error(pipe.predict(X_va), y_va))
    return float(np.mean(fold_maes))

for model_name, model_class, params, fixed_kwargs in model_configs:
    with mlflow.start_run(run_name=f"{model_name}_seed_variance") as parent_run:
        seed_maes = []
        for seed in seeds:
            cv_mae = cv_mae_for_seed(model_class, params, fixed_kwargs, seed)
            seed_maes.append(cv_mae)

            with mlflow.start_run(run_name=f"{model_name}_seed{seed}", nested=True):
                mlflow.log_params({**params, **fixed_kwargs, "random_state": seed})
                mlflow.log_metric("cv_mae", cv_mae)

        mean_mae = float(np.mean(seed_maes))
        std_mae  = float(np.std(seed_maes))

        mlflow.log_params(params)
        mlflow.log_metric("mean_cv_mae", mean_mae)
        mlflow.log_metric("std_cv_mae",  std_mae)
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

    print(f"{model_name}: mean_cv_mae={mean_mae:.4f} ± {std_mae:.4f} MW")




mlflow.set_experiment("CV-Strategy-Comparison")


cv_strategies = {
    "TimeSeriesSplit": TimeSeriesSplit(n_splits=5),
    "RandomKFold":      KFold(n_splits=5, shuffle=True, random_state=42),
}

with mlflow.start_run(run_name="RF_cv_strategy_comparison") as parent_run:
    strategy_maes = {}

    for strategy_name, cv in cv_strategies.items():
        fold_maes = []
        for train_idx, val_idx in cv.split(joined_dfs):
            X_tr = joined_dfs.iloc[train_idx]
            y_tr = joined_dfs["Total"].iloc[train_idx]
            X_va = joined_dfs.iloc[val_idx]
            y_va = joined_dfs["Total"].iloc[val_idx]

            pipe = Pipeline(_preprocessing + [
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
    ax.set_title("CV strategy comparison — leakage check")
    plt.tight_layout()
    mlflow.log_figure(fig, "cv_strategy_comparison.png")
    plt.close(fig)

print(f"TimeSeriesSplit MAE: {strategy_maes['TimeSeriesSplit']:.4f} MW")
print(f"Random KFold MAE:    {strategy_maes['RandomKFold']:.4f} MW")
print(f"Gap (negative = leakage signal): {leakage_gap:.4f} MW")
"""
Shared code for the wind-power-forecast MLflow project.

Every entry point (train.py, rf_gridsearch.py, mlp_gridsearch.py, baseline.py,
seed_variance.py, cv_comparison.py) imports from here rather than redefining
the preprocessing pipeline, data loading, or pipeline construction logic.
"""

import numpy as np
import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer


# ── Wind direction -> degrees ────────────────────────────────────────────────
deg_conversion = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
}


class MLflowJSONSanitizer(BaseEstimator, TransformerMixin):
    """Normalises raw or JSON-decoded input into a time-indexed DataFrame."""

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
    """Drops every column except Speed, Direction, and the helper flag."""

    _KEEP = ['Speed', 'Direction', '_from_wind_df']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = [c for c in self._KEEP if c in df.columns]
        return df[cols]


class WindVectorEncoder(BaseEstimator, TransformerMixin):
    """Encodes Direction + Speed into vector components (speed_u, speed_v)."""

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


PREPROCESSING = [
    ("sanitizer",    MLflowJSONSanitizer()),
    ("selector",     FeatureSelector()),
    ("speed_vector", WindVectorEncoder()),
    ("imputer",      SimpleImputer(strategy="median")),
    ("scaler",       StandardScaler()),
]


def build_pipelines():
    """Returns (pipeline_lr, pipeline_rf, pipeline_nn) with the project's default hyperparameters."""
    pipeline_lr = Pipeline(PREPROCESSING + [
        ("model", LinearRegression()),
    ])
    pipeline_rf = Pipeline(PREPROCESSING + [
        ("model", RandomForestRegressor(max_depth=5, n_jobs=-1)),
    ])
    pipeline_nn = Pipeline(PREPROCESSING + [
        ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="tanh", alpha=0.001,
            learning_rate_init=0.01, max_iter=600,
        )),
    ])
    return pipeline_lr, pipeline_rf, pipeline_nn


def load_data(power_path="data/power.csv", weather_path="data/weather.csv"):
    """Loads power + weather CSVs, aligns them onto a 3-hourly grid, returns the joined DataFrame."""
    power_df = pd.read_csv(power_path, parse_dates=["time"])
    wind_df = pd.read_csv(weather_path, parse_dates=["time"])
    power_df = power_df.drop(columns=[c for c in ['ANM', 'Non-ANM'] if c in power_df.columns])

    power_3h = power_df.set_index('time').resample('3h').mean().reset_index()
    joined_dfs = pd.merge_asof(
        power_3h, wind_df, on='time', direction='backward', tolerance=pd.Timedelta('3h')
    ).dropna()
    return joined_dfs


def split_train_test(joined_dfs, train_frac=0.8):
    """80/20 chronological split (no shuffling) — preserves temporal order."""
    X = joined_dfs.drop(columns=['Total'])
    y = joined_dfs['Total']
    split_idx = int(len(joined_dfs) * train_frac)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def extract_model_params(pipeline, exclude=("random_state", "n_jobs")):
    """Pulls hyperparameters from a fitted/unfitted pipeline's model step."""
    model_params = pipeline.named_steps["model"].get_params()
    return {k: v for k, v in model_params.items() if k not in exclude}


def set_tracking_uri():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

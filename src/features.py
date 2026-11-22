"""Shared feature definitions + preprocessing pipeline (used by train and app)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["sqft", "bedrooms", "bathrooms", "year_built", "lot_size", "garage_spaces",
           "overall_quality", "has_pool", "dist_to_center_km", "house_age",
           "sqft_per_bedroom", "bath_bed_ratio"]
CATEGORICAL = ["neighborhood"]
RAW_FEATURES = ["sqft", "bedrooms", "bathrooms", "year_built", "lot_size", "garage_spaces",
                "overall_quality", "has_pool", "dist_to_center_km", "neighborhood"]
NEIGHBORHOODS = ["Downtown", "Riverside", "Greenfield", "Oakwood", "Eastgate", "Millbrook"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds house_age, sqft_per_bedroom, bath_bed_ratio."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["house_age"] = 2024 - X["year_built"]
        X["sqft_per_bedroom"] = X["sqft"] / X["bedrooms"].clip(lower=1)
        X["bath_bed_ratio"] = X["bathrooms"] / X["bedrooms"].clip(lower=1)
        return X


def build_preprocessor() -> Pipeline:
    ct = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("fe", FeatureEngineer()), ("ct", ct)])

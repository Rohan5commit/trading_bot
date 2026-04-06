from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

LABEL_MAP = {
    "STRONG_SELL": 0,
    "SELL": 1,
    "NEUTRAL": 2,
    "BUY": 3,
    "STRONG_BUY": 4,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/corpus/tabular.parquet")
    parser.add_argument("--out", default="artifacts/tabular_model.ubj")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    df = df.dropna(subset=["label"])
    df["label_id"] = df["label"].map(LABEL_MAP)

    feature_cols = [c for c in df.columns if c not in ("label", "label_id")]
    X = df[feature_cols].select_dtypes(include=[np.number]).values
    y = df["label_id"].values

    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "multi:softprob",
        "num_class": 5,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    model = xgb.train(params, dtrain, num_boost_round=200)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.out)

    with open(Path(args.out).parent / "tabular_metadata.json", "w") as f:
        json.dump({"features": feature_cols}, f, indent=2)


if __name__ == "__main__":
    main()

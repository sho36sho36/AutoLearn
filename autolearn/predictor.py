from __future__ import annotations

from pathlib import Path

import pandas as pd
import joblib

from .analyzer import load_csv


def load_model(path: str | Path):
    """保存済みモデルを読み込む。"""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"モデルが見つかりません: {path}"
        )

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError(
            "有効なAutoLearnモデルではありません。"
        )

    required = {
        "version",
        "task",
        "target",
        "model_name",
        "model",
        "features",
    }

    missing = required - set(artifact)

    if missing:
        raise ValueError(
            f"モデル情報が不足しています: {missing}"
        )

    return artifact


def predict(
    model_path: str | Path,
    csv_path: str | Path,
    output: str | Path | None = None,
):
    """CSVから予測する。"""

    artifact = load_model(model_path)

    df = load_csv(csv_path)

    target = artifact["target"]
    features = artifact["features"]
    model = artifact["model"]

    missing_features = [
        column
        for column in features
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "必要な特徴量がCSVにありません: "
            + ", ".join(missing_features)
        )

    X = df[features]

    predictions = model.predict(X)

    result = df.copy()

    result[f"prediction_{target}"] = predictions

    if output is not None:
        output = Path(output)
        result.to_csv(
            output,
            index=False,
            encoding="utf-8-sig",
        )

    return result
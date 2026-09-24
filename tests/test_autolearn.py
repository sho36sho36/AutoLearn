import tempfile
from pathlib import Path

import pandas as pd

from autolearn.analyzer import analyze_dataframe
from autolearn.predictor import load_model, predict
from autolearn.trainer import (
    infer_task,
    save_model,
    train_models,
)


def test_analyzer():
    df = pd.DataFrame(
        {
            "age": [10, 20, 30],
            "name": ["A", "B", "C"],
        }
    )

    info = analyze_dataframe(df)

    assert info["rows"] == 3
    assert info["columns"] == 2


def test_infer_classification():
    y = pd.Series(
        ["A", "B", "A", "B"]
    )

    assert infer_task(y) == "classification"


def test_infer_regression():
    y = pd.Series(
        [10.2, 20.3, 30.4, 40.5]
    )

    assert infer_task(y) == "regression"


def test_training_and_prediction():
    df = pd.DataFrame(
        {
            "x1": range(20),
            "x2": [x * 2 for x in range(20)],
            "target": [
                "A" if x < 10 else "B"
                for x in range(20)
            ],
        }
    )

    info = train_models(
        df,
        target="target",
        task="classification",
        cv=2,
    )

    assert info["model"] is not None
    assert info["model_name"]

    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "model.joblib"
        csv_path = Path(directory) / "data.csv"

        save_model(
            info,
            model_path,
        )

        df.drop(
            columns=["target"]
        ).to_csv(
            csv_path,
            index=False,
        )

        artifact = load_model(
            model_path
        )

        assert artifact["target"] == "target"

        result = predict(
            model_path,
            csv_path,
        )

        assert len(result) == 20
        assert "prediction_target" in result.columns
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from .models import get_models
from .preprocessing import build_preprocessor


def infer_task(y) -> str:
    """目的変数から回帰・分類を推定する。"""

    # 文字列・カテゴリ・boolは分類
    if (
        y.dtype == "object"
        or str(y.dtype) == "category"
        or str(y.dtype) == "bool"
    ):
        return "classification"

    unique_count = y.nunique(dropna=True)
    sample_count = len(y)

    # 数値データ
    if np.issubdtype(y.dtype, np.number):

        # ユニーク値の割合が低く、かつ整数なら分類候補
        unique_ratio = unique_count / sample_count

        if (
            np.issubdtype(y.dtype, np.integer)
            and unique_count <= 20
            and unique_ratio <= 0.2
        ):
            return "classification"

        return "regression"

    return "regression"


def build_pipeline(X, model):
    """前処理＋モデルのPipelineを作成する。"""

    preprocessor = build_preprocessor(X)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _score_model(
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    task,
):
    """モデルを学習してテストデータで評価する。"""

    pipeline.fit(X_train, y_train)

    prediction = pipeline.predict(X_test)

    if task == "classification":
        accuracy = accuracy_score(
            y_test,
            prediction,
        )

        f1 = f1_score(
            y_test,
            prediction,
            average="weighted",
            zero_division=0,
        )

        return {
            "primary": accuracy,
            "accuracy": accuracy,
            "f1": f1,
        }

    mse = mean_squared_error(
        y_test,
        prediction,
    )

    rmse = float(np.sqrt(mse))

    mae = mean_absolute_error(
        y_test,
        prediction,
    )

    r2 = r2_score(
        y_test,
        prediction,
    )

    return {
        "primary": r2,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
    }


def train_models(
    df,
    target: str,
    task: str = "auto",
    cv: int = 5,
    test_size: float = 0.2,
):
    """複数モデルを学習・比較する。"""

    if target not in df.columns:
        raise ValueError(
            f"目的変数 '{target}' が見つかりません。"
        )

    if len(df) < 10:
        raise ValueError(
            "データが少なすぎます。10行以上を推奨します。"
        )

    data = df.dropna(subset=[target]).copy()

    if data.empty:
        raise ValueError(
            "目的変数に有効なデータがありません。"
        )

    X = data.drop(columns=[target])
    y = data[target]

    if task == "auto":
        task = infer_task(y)

    if task not in ("classification", "regression"):
        raise ValueError(
            "taskは auto / classification / regression "
            "のいずれかです。"
        )

    if X.shape[1] == 0:
        raise ValueError(
            "特徴量がありません。"
        )

    if task == "classification":
        if y.nunique() < 2:
            raise ValueError(
                "分類には2クラス以上が必要です。"
            )

        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    models = get_models(task)

    if task == "classification":
        splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=42,
        )

        scoring = "accuracy"

    else:
        splitter = KFold(
            n_splits=cv,
            shuffle=True,
            random_state=42,
        )

        scoring = "r2"

    results = []
    successful = []

    print()
    print("=" * 70)
    print("🤖 AutoLearn")
    print("=" * 70)

    print(f"タスク       : {task}")
    print(f"データ数     : {len(data):,}")
    print(f"特徴量数     : {X.shape[1]}")
    print(f"モデル数     : {len(models)}")
    print(f"Cross Validation : {cv}-fold")

    print()
    print("🔍 モデルを比較しています...")
    print()

    for name, model in models.items():
        start = time.perf_counter()

        try:
            pipeline = build_pipeline(
                X_train,
                clone(model),
            )

            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=splitter,
                scoring=scoring,
                n_jobs=1,
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            metrics = _score_model(
                pipeline,
                X_train,
                X_test,
                y_train,
                y_test,
                task,
            )

            elapsed = time.perf_counter() - start

            result = {
                "model": name,
                "cv_mean": float(scores.mean()),
                "cv_std": float(scores.std()),
                "test_score": float(metrics["primary"]),
                "time": elapsed,
                "status": "OK",
                **metrics,
            }

            results.append(result)
            successful.append(
                (name, pipeline, result)
            )

            if task == "classification":
                print(
                    f"✓ {name:<35}"
                    f"CV={scores.mean():.4f}  "
                    f"Test={metrics['accuracy']:.4f}"
                )
            else:
                print(
                    f"✓ {name:<35}"
                    f"CV={scores.mean():.4f}  "
                    f"Test R²={metrics['r2']:.4f}"
                )

        except Exception as exc:
            elapsed = time.perf_counter() - start

            results.append(
                {
                    "model": name,
                    "cv_mean": None,
                    "cv_std": None,
                    "test_score": None,
                    "time": elapsed,
                    "status": "FAILED",
                    "error": str(exc),
                }
            )

            print(
                f"✗ {name:<35}"
                f"利用不可: {str(exc)[:60]}"
            )

    if not successful:
        raise RuntimeError(
            "利用可能なモデルが1つもありませんでした。"
        )

    # CV平均を基準に選択
    successful.sort(
        key=lambda item: item[2]["cv_mean"],
        reverse=True,
    )

    best_name, best_pipeline, best_result = (
        successful[0]
    )

    print()
    print("=" * 70)
    print("🏆 選択されたモデル")
    print("=" * 70)

    print(f"モデル : {best_name}")
    print(
        f"CV     : "
        f"{best_result['cv_mean']:.4f}"
    )

    if task == "classification":
        print(
            f"Accuracy : "
            f"{best_result['accuracy']:.4f}"
        )
        print(
            f"F1       : "
            f"{best_result['f1']:.4f}"
        )
    else:
        print(
            f"R²  : "
            f"{best_result['r2']:.4f}"
        )
        print(
            f"MAE : "
            f"{best_result['mae']:.4f}"
        )
        print(
            f"RMSE: "
            f"{best_result['rmse']:.4f}"
        )

    return {
        "task": task,
        "target": target,
        "model_name": best_name,
        "model": best_pipeline,
        "results": results,
        "features": list(X.columns),
    }


def save_model(
    model_info: dict,
    path: str | Path = "model.joblib",
):
    """モデルとメタデータを保存する。"""

    path = Path(path)

    artifact = {
        "version": "1.0.0",
        "task": model_info["task"],
        "target": model_info["target"],
        "model_name": model_info["model_name"],
        "model": model_info["model"],
        "features": model_info["features"],
        "results": model_info["results"],
    }

    joblib.dump(
        artifact,
        path,
    )

    return path
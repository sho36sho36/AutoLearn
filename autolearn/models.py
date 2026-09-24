from __future__ import annotations

from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from sklearn.svm import (
    LinearSVC,
    LinearSVR,
    NuSVC,
    NuSVR,
    SVC,
    SVR,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)


def get_classification_models():
    """分類モデル一覧を返す。"""

    return {
        "LogisticRegression":
            LogisticRegression(max_iter=2000),

        "RidgeClassifier":
            RidgeClassifier(),

        "SVC":
            SVC(),

        "LinearSVC":
            LinearSVC(max_iter=5000),

        "NuSVC":
            NuSVC(),

        "KNeighborsClassifier":
            KNeighborsClassifier(),

        "DecisionTreeClassifier":
            DecisionTreeClassifier(
                random_state=42
            ),

        "ExtraTreeClassifier":
            ExtraTreeClassifier(
                random_state=42
            ),

        "RandomForestClassifier":
            RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                n_jobs=-1,
            ),

        "ExtraTreesClassifier":
            ExtraTreesClassifier(
                n_estimators=150,
                random_state=42,
                n_jobs=-1,
            ),

        "GradientBoostingClassifier":
            GradientBoostingClassifier(
                random_state=42
            ),

        "HistGradientBoostingClassifier":
            HistGradientBoostingClassifier(
                random_state=42
            ),

        "AdaBoostClassifier":
            AdaBoostClassifier(
                random_state=42
            ),

        "BaggingClassifier":
            BaggingClassifier(
                random_state=42,
                n_jobs=-1,
            ),

        "GaussianProcessClassifier":
            GaussianProcessClassifier(
                kernel=RBF()
            ),
    }


def get_regression_models():
    """回帰モデル一覧を返す。"""

    return {
        "LinearRegression":
            LinearRegression(),

        "Ridge":
            Ridge(),

        "ElasticNet":
            ElasticNet(
                random_state=42
            ),

        "SVR":
            SVR(),

        "LinearSVR":
            LinearSVR(
                max_iter=5000,
                random_state=42,
            ),

        "NuSVR":
            NuSVR(),

        "KNeighborsRegressor":
            KNeighborsRegressor(),

        "DecisionTreeRegressor":
            DecisionTreeRegressor(
                random_state=42
            ),

        "ExtraTreeRegressor":
            ExtraTreeRegressor(
                random_state=42
            ),

        "RandomForestRegressor":
            RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                n_jobs=-1,
            ),

        "ExtraTreesRegressor":
            ExtraTreesRegressor(
                n_estimators=150,
                random_state=42,
                n_jobs=-1,
            ),

        "GradientBoostingRegressor":
            GradientBoostingRegressor(
                random_state=42
            ),

        "HistGradientBoostingRegressor":
            HistGradientBoostingRegressor(
                random_state=42
            ),

        "AdaBoostRegressor":
            AdaBoostRegressor(
                random_state=42
            ),

        "BaggingRegressor":
            BaggingRegressor(
                random_state=42,
                n_jobs=-1,
            ),

        "GaussianProcessRegressor":
            GaussianProcessRegressor(
                kernel=RBF()
            ),
    }


def get_models(task: str):
    if task == "classification":
        return get_classification_models()

    if task == "regression":
        return get_regression_models()

    raise ValueError(
        f"未知のタスクです: {task}"
    )
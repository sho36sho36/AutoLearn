# 🤖 AutoLearn

CSVを渡すだけで機械学習モデルを自動比較・選択する、軽量なAutoML風CLIツールです。

scikit-learnを利用して、データ分析、前処理、複数モデルの学習、Cross Validation、モデル選択、予測までを自動化します。

---

## ✨ Features

- 📊 CSVデータ自動分析
- 🤖 複数の機械学習モデルを自動比較
- 🔮 回帰・分類に対応
- 🧹 欠損値を自動処理
- 🏷️ カテゴリ変数を自動処理
- ⚙️ scikit-learn Pipelineを使用
- 🔀 Cross Validation対応
- 🏆 モデルをCVスコアで比較
- 💾 学習済みモデルをjoblibで保存
- 🔮 保存したモデルから予測
- 💻 CLIで簡単に利用可能

---

## 📦 Installation

Python 3.10以上を推奨します。

~~~bash
pip install -r requirements.txt
~~~

開発環境としてインストールする場合:

~~~bash
pip install -e .
~~~

---

## 🚀 Usage

### データ分析

~~~bash
python -m autolearn analyze examples/iris.csv
~~~

データセットの行数、列数、データ型、欠損値、ユニーク数などを確認できます。

### 学習

~~~bash
python -m autolearn train examples/iris.csv --target target
~~~

モデルはデフォルトで `model.joblib` に保存されます。

### 回帰

~~~bash
python -m autolearn train examples/regression.csv --target price
~~~

### タスクを明示

自動判定ではなく、明示的に分類を指定できます。

~~~bash
python -m autolearn train data.csv --target target --task classification
~~~

回帰:

~~~bash
python -m autolearn train data.csv --target price --task regression
~~~

### Cross Validation

~~~bash
python -m autolearn train data.csv --target target --cv 10
~~~

### モデル保存先を指定

~~~bash
python -m autolearn train data.csv --target target --output my_model.joblib
~~~

---

## 🔮 Prediction

学習済みモデルを使って予測できます。

~~~bash
python -m autolearn predict model.joblib new_data.csv
~~~

結果をCSVに保存:

~~~bash
python -m autolearn predict model.joblib new_data.csv --output predictions.csv
~~~

予測結果には `prediction_目的変数名` という列が追加されます。

---

## 🤖 Models

### Classification

AutoLearn v1.0.0では以下の分類モデルを比較します。

- LogisticRegression
- RidgeClassifier
- SVC
- LinearSVC
- NuSVC
- KNeighborsClassifier
- DecisionTreeClassifier
- ExtraTreeClassifier
- RandomForestClassifier
- ExtraTreesClassifier
- GradientBoostingClassifier
- HistGradientBoostingClassifier
- AdaBoostClassifier
- BaggingClassifier
- GaussianProcessClassifier

### Regression

以下の回帰モデルを比較します。

- LinearRegression
- Ridge
- ElasticNet
- SVR
- LinearSVR
- NuSVR
- KNeighborsRegressor
- DecisionTreeRegressor
- ExtraTreeRegressor
- RandomForestRegressor
- ExtraTreesRegressor
- GradientBoostingRegressor
- HistGradientBoostingRegressor
- AdaBoostRegressor
- BaggingRegressor
- GaussianProcessRegressor

モデルの学習に失敗した場合、そのモデルは自動的にスキップされ、利用可能なモデルだけで比較を続行します。

---

## ⚙️ Automatic Machine Learning

AutoLearnは以下の処理を自動化します。

~~~text
CSV
 ↓
データ読み込み
 ↓
データ分析
 ↓
回帰 / 分類の判定
 ↓
特徴量の確認
 ↓
自動前処理
 ↓
複数モデルの学習
 ↓
Cross Validation
 ↓
モデル比較
 ↓
最適モデル選択
 ↓
モデル保存
~~~

---

## 🧹 Automatic Preprocessing

AutoLearnは列の型を確認して、自動的に前処理を構築します。

### 数値列

~~~text
欠損値
 ↓
中央値で補完
 ↓
StandardScaler
~~~

### カテゴリ列

~~~text
欠損値
 ↓
最頻値で補完
 ↓
OneHotEncoder
~~~

前処理とモデルはscikit-learnのPipelineとして一体化されます。

そのため、学習時と予測時で同じ前処理を使用できます。

---

## 📊 Evaluation

### Classification

主な評価指標:

- Accuracy
- F1 Score

モデル選択にはCross ValidationのAccuracyを使用します。

### Regression

主な評価指標:

- R²
- MAE
- RMSE

モデル選択にはCross ValidationのR²を使用します。

---

## 🔬 Cross Validation

AutoLearnではデフォルトで5-fold Cross Validationを使用します。

~~~text
Training Data
      │
      ├── Fold 1
      ├── Fold 2
      ├── Fold 3
      ├── Fold 4
      └── Fold 5
             ↓
       平均スコアを計算
~~~

分割数は `--cv` で変更できます。

~~~bash
python -m autolearn train data.csv --target target --cv 10
~~~

---

## 📈 Example Output

分類の場合:

~~~text
======================================================================
🤖 AutoLearn
======================================================================
タスク       : classification
データ数     : 150
特徴量数     : 4
モデル数     : 15
Cross Validation : 5-fold

🔍 モデルを比較しています...

✓ LogisticRegression                 CV=0.9500  Test=0.9667
✓ SVC                                CV=0.9583  Test=0.9667
✓ KNeighborsClassifier               CV=0.9500  Test=1.0000
✓ RandomForestClassifier             CV=0.9417  Test=0.9333
✓ ExtraTreesClassifier               CV=0.9500  Test=0.9667
...

======================================================================
🏆 選択されたモデル
======================================================================
モデル : SVC
CV     : 0.9583
Accuracy : 0.9667
F1       : 0.9666
======================================================================

======================================================================
💾 モデル保存完了
======================================================================
ファイル : model.joblib
モデル   : SVC
タスク   : classification
======================================================================
~~~

※実際のスコアはデータセット、分割方法、scikit-learnのバージョンなどによって変化します。

---

## ⚠️ Notes

AutoLearnはモデル選択を自動化するためのツールです。

Cross Validationやテストデータの結果は、データセットによって大きく変わります。

特に小規模なデータセットでは、評価値が不安定になる場合があります。

AutoLearnが選択したモデルが、すべての未知データで最も良い結果になることを保証するものではありません。

重要な用途では、データの品質、評価方法、モデルの妥当性などを確認してください。

---

## 🧪 Tests

テストを実行:

~~~bash
pytest
~~~

または:

~~~bash
python -m pytest
~~~

---

## 📁 Project Structure

~~~text
AutoLearn/
├── autolearn/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── analyzer.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── trainer.py
│   └── predictor.py
│
├── examples/
│   ├── iris.csv
│   └── regression.csv
│
├── tests/
│   └── test_autolearn.py
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
└── .gitignore
~~~

---

## 🛠️ Requirements

- Python 3.10+
- pandas
- NumPy
- scikit-learn
- joblib

---

## 📜 License

MIT License

---

## 🎯 Version

Current version:

**v1.0.0**

AutoLearn v1.0.0 is a lightweight AutoML-style tool built with Python and scikit-learn.
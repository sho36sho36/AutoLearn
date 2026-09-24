from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """CSVファイルを読み込む。"""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSVファイルが見つかりません: {path}")

    if not path.is_file():
        raise ValueError(f"ファイルではありません: {path}")

    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp932")
    except Exception as exc:
        raise ValueError(f"CSVを読み込めませんでした: {exc}") from exc

    if df.empty:
        raise ValueError("CSVにデータがありません。")

    return df


def analyze_dataframe(df: pd.DataFrame) -> dict:
    """DataFrameの基本的な情報を分析する。"""

    numeric_columns = list(
        df.select_dtypes(include="number").columns
    )

    categorical_columns = list(
        df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
    )

    missing = int(df.isna().sum().sum())
    duplicated = int(df.duplicated().sum())

    columns = []

    for column in df.columns:
        series = df[column]

        columns.append(
            {
                "name": column,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
            }
        )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": missing,
        "duplicated_rows": duplicated,
        "column_details": columns,
    }


def print_analysis(df: pd.DataFrame) -> None:
    """分析結果をCLIに表示する。"""

    info = analyze_dataframe(df)

    print()
    print("=" * 70)
    print("📊 データセット分析")
    print("=" * 70)

    print(f"行数           : {info['rows']:,}")
    print(f"列数           : {info['columns']:,}")
    print(f"数値列         : {len(info['numeric_columns'])}")
    print(f"カテゴリ列     : {len(info['categorical_columns'])}")
    print(f"欠損値         : {info['missing_values']:,}")
    print(f"重複行         : {info['duplicated_rows']:,}")

    print()
    print("列情報")
    print("-" * 70)

    for column in info["column_details"]:
        print(
            f"{column['name']:<25}"
            f"type={column['dtype']:<12}"
            f"missing={column['missing']:<6}"
            f"unique={column['unique']}"
        )

    print("=" * 70)
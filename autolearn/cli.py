from __future__ import annotations

import argparse
import sys

from .analyzer import load_csv, print_analysis
from .predictor import predict
from .trainer import save_model, train_models


def build_parser():
    parser = argparse.ArgumentParser(
        prog="autolearn",
        description=(
            "🤖 AutoLearn - "
            "CSVから機械学習を自動化するAutoML風ツール"
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # analyze
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="CSVデータを分析する",
    )

    analyze_parser.add_argument(
        "csv",
        help="分析するCSVファイル",
    )

    # train
    train_parser = subparsers.add_parser(
        "train",
        help="CSVからモデルを自動学習する",
    )

    train_parser.add_argument(
        "csv",
        help="学習用CSV",
    )

    train_parser.add_argument(
        "--target",
        required=True,
        help="目的変数の列名",
    )

    train_parser.add_argument(
        "--task",
        choices=[
            "auto",
            "classification",
            "regression",
        ],
        default="auto",
        help="タスク種別",
    )

    train_parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Cross Validationの分割数",
    )

    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="テストデータ割合",
    )

    train_parser.add_argument(
        "--output",
        default="model.joblib",
        help="保存するモデルファイル",
    )

    # predict
    predict_parser = subparsers.add_parser(
        "predict",
        help="保存済みモデルで予測する",
    )

    predict_parser.add_argument(
        "model",
        help="model.joblib",
    )

    predict_parser.add_argument(
        "csv",
        help="予測対象CSV",
    )

    predict_parser.add_argument(
        "--output",
        default=None,
        help="予測結果CSV",
    )

    return parser


def command_train(args):
    df = load_csv(args.csv)

    info = train_models(
        df=df,
        target=args.target,
        task=args.task,
        cv=args.cv,
        test_size=args.test_size,
    )

    path = save_model(
        info,
        args.output,
    )

    print()
    print("=" * 70)
    print("💾 モデル保存完了")
    print("=" * 70)
    print(f"ファイル : {path}")
    print(f"モデル   : {info['model_name']}")
    print(f"タスク   : {info['task']}")
    print("=" * 70)


def command_predict(args):
    result = predict(
        model_path=args.model,
        csv_path=args.csv,
        output=args.output,
    )

    print()
    print("=" * 70)
    print("🔮 予測完了")
    print("=" * 70)

    print(result.to_string(index=False))

    if args.output:
        print()
        print(f"💾 保存先: {args.output}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "analyze":
            df = load_csv(args.csv)
            print_analysis(df)

        elif args.command == "train":
            command_train(args)

        elif args.command == "predict":
            command_predict(args)

    except KeyboardInterrupt:
        print("\n\n⚠️ 処理を中断しました。")
        sys.exit(130)

    except Exception as exc:
        print()
        print(f"❌ エラー: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
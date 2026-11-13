"""Command line interface. Training requires the explicit 'train' command."""

import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline xArm cycle-level load classification")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Download/verify selected raw files and extract features; does not train")
    prep.add_argument("--source", required=True, type=Path, help="Local xArm-model-training Git clone")
    prep.add_argument("--out", required=True, type=Path)
    prep.add_argument("--cache", type=Path)
    training = commands.add_parser("train", help="Explicitly fit models and select on development dates only")
    training.add_argument("--data", required=True, type=Path)
    training.add_argument("--out", required=True, type=Path)
    training.add_argument("--config", type=Path)
    evaluation = commands.add_parser("evaluate", help="Evaluate saved models; does not train or tune")
    evaluation.add_argument("--model", required=True, type=Path)
    evaluation.add_argument("--data", required=True, type=Path)
    evaluation.add_argument("--out", required=True, type=Path)
    evaluation.add_argument("--no-reference", action="store_true", help="Skip archived upstream models")
    prediction = commands.add_parser("predict", help="Predict from a cycle-feature CSV; does not train")
    prediction.add_argument("--model", required=True, type=Path)
    prediction.add_argument("--features", required=True, type=Path)
    prediction.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from .prepare import prepare
            prepare(args.source, args.out, args.cache)
        elif args.command == "train":
            from .train import train
            train(args.data, args.out, args.config)
        elif args.command == "evaluate":
            from .evaluate import evaluate
            evaluate(args.model, args.data, args.out, references=not args.no_reference)
        else:
            from .io import read_csv
            from .model import load_bundle, predict_frame
            if args.out.exists():
                raise FileExistsError("Use a new prediction output file")
            result = predict_frame(load_bundle(args.model), read_csv(args.features))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8", newline="") as stream:
                result.to_csv(stream, index=False)
            print(f"Saved {len(result)} predictions to {args.out}")
    except (ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

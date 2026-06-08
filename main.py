import argparse
import logger_config
from process_data import split_data
from test import run_test
from train import run_training, resume_training


def main():
    parser = argparse.ArgumentParser(
        description="AttnGIN-DDI: Drug-Drug Interaction Prediction"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = sub.add_parser("train", help="Start a new training run")
    p_train.add_argument("name", help="Config name in config.py (e.g. default)")
    p_train.add_argument(
        "--encoder",
        choices=["AttnEncoder", "AttnResEncoder"],
        help="Encoder type (default: AttnEncoder)",
    )
    p_train.add_argument(
        "--metric-average",
        choices=["macro", "weighted", "micro"],
        dest="metric_average",
        help="Metric averaging strategy (default: macro)",
    )
    p_train.add_argument(
        "--data-source",
        choices=["drugbank", "twosides"],
        dest="data_source",
        help="Data source (default: drugbank)",
    )
    p_train.add_argument(
        "--split-type",
        choices=["random", "cluster"],
        dest="split_type",
        help="Split strategy (default: random)",
    )

    # ---- resume ----
    p_resume = sub.add_parser("resume", help="Resume a training run from checkpoint")
    p_resume.add_argument("name", help="Experiment name (directory under checkpoints/)")

    # ---- test ----
    p_test = sub.add_parser("test", help="Evaluate a trained model on test set")
    p_test.add_argument("name", help="Experiment name")

    # ---- split ----
    p_split = sub.add_parser("split", help="Split raw data into train/test")
    p_split.add_argument(
        "--data-source",
        default="drugbank",
        choices=["drugbank", "twosides"],
        dest="data_source",
        help="Data source (default: drugbank)",
    )
    p_split.add_argument(
        "--split-type",
        default="random",
        choices=["random", "cluster"],
        dest="split_type",
        help="Split strategy (default: random)",
    )
    p_split.add_argument(
        "--train-size",
        type=float,
        default=0.8,
        dest="train_size",
        help="Training set ratio (default: 0.8)",
    )
    p_split.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    if args.command == "train":
        run_training(
            name=args.name,
            encoder=args.encoder,
            metric_average=args.metric_average,
            data_source=args.data_source,
            split_type=args.split_type,
        )
    elif args.command == "resume":
        resume_training(name=args.name)
    elif args.command == "test":
        run_test(name=args.name)
    elif args.command == "split":
        split_data(
            data_source=args.data_source,
            split_type=args.split_type,
            train_size=args.train_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
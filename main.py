import argparse
from process_data import split_data
from test import run_testing,test_all
from train import run_training, resume_training, train_all



def main():
    parser = argparse.ArgumentParser(
        description="AttnGIN-DDI: Drug-Drug Interaction Prediction"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_train = sub.add_parser("train", help="Start a new training run")
    p_train.add_argument("config", help="Config class name in config.py")

    # ---- train all----
    p_train_all = sub.add_parser("train_all", help="Queue a training run")
    p_train_all.add_argument(
        "configs", nargs="+", help="Config class name(s) in config.py"
    )

    # ---- resume ----
    p_resume = sub.add_parser(
        "resume", help="Resume a training run that has a checkpoint under checkpoints/"
    )
    p_resume.add_argument("config", help="Config class name in config.py")

    # ---- test ----
    p_test = sub.add_parser("test", help="Evaluate a trained model on test set")
    p_test.add_argument("config", help="Config class name in config.py")


    # ---- test all----
    p_test_all = sub.add_parser("test_all", help="Queue a testing run")
    p_test_all.add_argument(
        "configs", nargs="+", help="Config class name(s) in config.py"
    )



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
        run_training(config_class_name=args.config)
    elif args.command == "train_all":
        train_all(args.configs)
    elif args.command == "resume":
        resume_training(config_class_name=args.config)
    elif args.command == "test":
        run_testing(config_class_name=args.config)
    elif args.command == "test_all":
        test_all(args.configs)
    elif args.command == "split":
        split_data(
            data_source=args.data_source,
            split_type=args.split_type,
            train_size=args.train_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

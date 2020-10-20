import argparse

from joeynmt.training import train
from joeynmt.prediction import test

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("mode", choices=["train", "test"],
                    help="train or test a model")

    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")

    args = ap.parse_args()

    if args.mode == "train":
        train(cfg_file=args.config_path)
    elif args.mode == "test":
        test(cfg_file=args.config_path, ckpt=args.ckpt,
             output_path=args.output_path, save_attention=args.save_attention)
    else:
        raise ValueError("Unknown mode")


if __name__ == "__main__":
    main()
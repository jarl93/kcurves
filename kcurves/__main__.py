import argparse
from data_management import load_data_set
from initialization import init_model
from training import train
from testing import test

def main():

    ap = argparse.ArgumentParser("kcurves")

    ap.add_argument("mode", choices=["train", "test"],
                    help="train or test a model")

    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")

    args = ap.parse_args()

    if args.mode == "train":
        data_set = load_data_set(cfg_path = args.config_path)
        model = init_model(cfg_path = args.config_path)
        train(cfg_path = args.config_path, model = model, data_set = data_set)

    elif args.mode == "test":
        data_set = load_data_set(cfg_path = args.config_path)
        # init_model is needed since in the training just the state_dict was saved
        model = init_model(cfg_path = args.config_path)
        test(cfg_path = args.config_path, model = model, data_set = data_set)

    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
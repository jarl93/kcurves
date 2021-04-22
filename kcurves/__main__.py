import argparse
import sys
from logger import Logger
from data_management import load_data_set
from initialization import init_model
from training import train
from testing import test
from helpers import load_config
import numpy as np

def main():

    ap = argparse.ArgumentParser("kcurves")

    ap.add_argument("mode", choices=["train", "test", "selection_hyperparameters"],
                    help="train or test a model or select hyperparameters")

    ap.add_argument("config_path", type=str,
                    help="path to YAML config file")

    args = ap.parse_args()

    # Set the Logger for the logging
    sys.stdout = Logger(cfg_path = args.config_path)

    if args.mode == "train":
        data_set = load_data_set(cfg_path = args.config_path)
        model = init_model(cfg_path = args.config_path)
        train(cfg_path = args.config_path, model = model, data_set = data_set)

    elif args.mode == "test":
        data_set = load_data_set(cfg_path = args.config_path)
        # init_model is needed since in the training just the state_dict was saved
        model = init_model(cfg_path = args.config_path)
        cfg_file = load_config(args.config_path)
        mode_forced = cfg_file["test"]["mode_forced"]
        _, _, _ = test(cfg_path=args.config_path, model=model, data_set=data_set, mode_forced=mode_forced, mode="final")

    elif args.mode == "selection_hyperparameters":
        list_ACC = []
        list_NMI = []
        list_ARI = []
        data_set = load_data_set(cfg_path = args.config_path)
        cfg_file = load_config(args.config_path)
        num_iterations = cfg_file["train"]["num_iterations"]
        for iter in range(num_iterations):
            print("Iteration: {}".format(iter+1))
            model = init_model(cfg_path = args.config_path)
            train(cfg_path = args.config_path, model = model, data_set = data_set)
            acc, nmi, ari = test(cfg_path=args.config_path, model=model, data_set=data_set,
                                 mode_forced="test", mode="final")
            list_ACC.append(acc)
            list_NMI.append(nmi)
            list_ARI.append(ari)

        avg_ACC = np.round(np.mean(list_ACC), 3)
        avg_NMI = np.round(np.mean(list_NMI), 3)
        avg_ARI = np.round(np.mean(list_ARI), 3)

        print("Average ACC = {}".format(avg_ACC))
        print("Average NMI = {}".format(avg_NMI))
        print("Average ARI = {}".format(avg_ARI))


    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
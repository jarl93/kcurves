from generate_config_files import generate_list
from helpers import create_dir
import os
import shutil
import argparse

def delete_dir(path):
    """
    Performs rm -r to a given path.
    :param path:
    :return: None
    """
    if os.path.isdir(path):
        shutil.rmtree(path)

    return None

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--all', action='store_true',
                    help='Clean all the directories including the config files.')

    ap.add_argument('-cfg', '--configs_generation', action='store_true',
                    help='Clean the config files for the generation process.')

    ap.add_argument('-cft', '--configs_training', action='store_true',
                    help='Clean the config files for training and testing.')

    ap.add_argument('-m', '--models', action='store_true',
                    help='Clean the directories of the models and its respective plots.')

    ap.add_argument('-p', '--plots', action='store_true',
                    help='Clean the plots of the data generated.')

    ap.add_argument('-d', '--data', action='store_true',
                    help='Clean the training and test data.')

    args = ap.parse_args()

    NUM_TESTS = 20

    config_path ="./configs/synthetic_clusters/"
    config_generation_path = "./configs/synthetic_generation_clusters/"

    train_path = "./data/synthetic_clusters/train/"
    test_path = "./data/synthetic_clusters/test/"
    plot_path = "./data/synthetic_clusters/plots/"
    model_path = "./models/synthetic_clusters/"

    if args.configs_generation or args.all:
        delete_dir(config_generation_path)
    if args.configs_training or args.all:
        delete_dir(config_path)

    list_train_paths = generate_list(train_path, NUM_TESTS, "/")
    list_test_paths = generate_list(test_path, NUM_TESTS, "/")
    list_plot_paths = generate_list(plot_path, NUM_TESTS, "/")
    list_model_paths = generate_list(model_path, NUM_TESTS, "/")

    for i in range(1, NUM_TESTS+1):
        if args.data or args.all:
            delete_dir(list_train_paths[i])
            delete_dir(list_test_paths[i])
            create_dir(list_train_paths[i])
            create_dir(list_test_paths[i])
        if args.plots or args.all:
            delete_dir(list_plot_paths[i])
            create_dir(list_plot_paths[i])
        if args.models or args.all:
            delete_dir(list_model_paths[i])
            create_dir(list_model_paths[i])


if __name__ == "__main__":
    main()
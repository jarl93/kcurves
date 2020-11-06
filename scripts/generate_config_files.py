from helpers import create_dir
import numpy as np
import argparse

def generate_list(str_, num_tests, char):
    """
    Add documentation.
    :param str_:
    :param num_tests:
    :param char:
    :return:
    """
    list_str = [None]
    for i in range(1, num_tests+1):
        num = ""
        # put 0's at the beginning w.r.t. num_tests
        dif = int(np.log10(num_tests)) - int(np.log10(i))
        for j in range(dif):
            num += "0"
        # add the number given by the iterator
        num += str(i)
        # handle cases depending on the char
        if char == "_":
            list_str.append(str_ + "_" + num)
        else:
            list_str.append(str_ + num + "/")
    return list_str

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--all', action='store_true',
                    help='Generate config files for the generation process and for training (and testing).')

    ap.add_argument('-g', '--generation', action='store_true',
                    help='Generate config files for the generation process.')

    ap.add_argument('-t', '--training', action='store_true',
                    help='Generate config files for training and testing.')

    args = ap.parse_args()

    NUM_TESTS = 20

    # create directory for the config files
    path_config_files = "./configs/synthetic_clusters/"
    create_dir(path_config_files)

    # main paths for training data, test data and plots (generated data)
    train_path = "./data/synthetic_clusters/train/"
    test_path = "./data/synthetic_clusters/test/"
    plot_path = "./data/synthetic_clusters/plots/"

    # also equal to number of classes
    num_centers = 2

    # code to generate config files that generate data
    if args.generation or args.all:

        # create directory for the config generation files
        path_config_generation_files = "./configs/synthetic_generation_clusters/"
        create_dir(path_config_generation_files)
        name = "synthetic_generation_clusters"
        # common hyperparameters
        # TODO: Consider to make lists for the hyperparameters

        dim = 2
        center_box = [-100, 100]
        cluster_std = 5
        scale_factor = 10
        num_samples_train = 5000
        num_samples_test = 1000

        # lists with the paths for the training data, test data and plots
        list_names = generate_list(name, NUM_TESTS, "_")
        list_train_paths = generate_list(train_path, NUM_TESTS, "/")
        list_test_paths = generate_list(test_path, NUM_TESTS, "/")
        list_plot_paths = generate_list(plot_path, NUM_TESTS, "/")

        for i in range(1, NUM_TESTS+1):
            random_state = i
            path = path_config_generation_files + list_names[i] + ".yaml"
            f = open(path, "w")
            # write the files
            f.write("name: " + list_names[i] + "\n")
            f.write("clusters:\n")
            f.write("  num_centers: " + str(num_centers) + "\n")
            f.write("  center_box: " + str(center_box) + "\n")
            f.write("  cluster_std: " + str(cluster_std) + "\n")
            f.write("  random_state: " + str(random_state) + "\n")
            f.write("  dim: " + str(dim) + "\n")
            f.write("data:\n")
            f.write("  save: True\n")
            f.write("  plot: True\n")
            f.write("  normalize: False\n")
            f.write("  scale: True\n")
            f.write("  scale_factor: " + str(scale_factor) + "\n")
            f.write("  train:\n")
            f.write("    path: " + list_train_paths[i] + "\n")
            f.write("    num_samples: " + str(num_samples_train) + "\n")
            f.write("  test:\n")
            f.write("    path: " + list_test_paths[i] + "\n")
            f.write("    num_samples: " + str(num_samples_test) + "\n")
            f.write("  plots:\n")
            f.write("    path: " + list_plot_paths[i] + "\n")
            f.close()

    # code to generate config files for training and testing
    if args.training or args.all:

        # create directory for the config files
        path_config_files = "./configs/synthetic_clusters/"
        create_dir(path_config_files)
        name = "synthetic_clusters"

        list_names = generate_list(name, NUM_TESTS, "_")
        list_train_paths = generate_list(train_path, NUM_TESTS, "/")
        list_test_paths = generate_list(test_path, NUM_TESTS, "/")
        model_name = "model_clusters"
        model_path = "./models/synthetic_clusters/"
        list_model_names = generate_list(model_name, NUM_TESTS, "_")
        list_model_paths = generate_list(model_path, NUM_TESTS, "/")

        # common hyperparameters
        # TODO: Consider to make lists for the hyperparameters

        # train:
        batch_size = 128
        num_epochs = 100
        alpha_min = 0.01
        alpha_max = 0.5
        beta_ = 0.05
        lambda_ = 0.0001
        lr = 0.001
        batch_frequency_loss = 1
        # test:
        mode_forced = True
        batch_size = 128

        # model:
        layer_sizes_encoder = [2]
        layer_sizes_decoder = [2]
        input_dim = 2
        latent_dim = 2

        #tracing:
        batch_frequency = 10

        for i in range(1, NUM_TESTS+1):
            path = path_config_files + list_names[i] + ".yaml"
            f = open(path, "w")
            # write the files
            f.write("name: " + list_names[i] + "\n")
            f.write("data:\n")
            f.write("  data_set: synthetic_clusters\n")
            f.write("  train: " + str(list_train_paths[i]) + "\n" )
            f.write("  test: " + str(list_test_paths[i]) + "\n")
            f.write("  num_classes: " + str(num_centers) + "\n")

            f.write("train:\n")
            f.write("  batch_size: " + str(batch_size) + "\n")
            f.write("  num_epochs: " + str(num_epochs) + "\n")
            f.write("  alpha_min: " + str(alpha_min) + "\n")
            f.write("  alpha_max: " + str(alpha_max) + "\n")
            f.write("  beta: " + str(beta_) + "\n")
            f.write("  lambda: " + str(lambda_) + "\n")
            f.write("  lr: " + str(lr) + "\n")
            f.write("  batch_frequency_loss: "+ str(batch_frequency_loss) + "\n")

            f.write("test:\n")
            f.write("  mode_forced: " + str(mode_forced) + "\n")
            f.write("  batch_size: " + str(batch_size) + "\n")

            f.write("model:\n")
            f.write("  path: " + list_model_paths[i] + "\n")
            f.write("  name: " + list_model_names[i] + "\n")
            f.write("  save: True\n")
            f.write("  device: cpu\n")
            f.write("  encoder:\n")
            f.write("    layer_sizes: " + str(layer_sizes_encoder) + "\n" )
            f.write("    last_nn_layer: Identity\n")
            f.write("  decoder:\n")
            f.write("    layer_sizes: " + str(layer_sizes_decoder) + "\n")
            f.write("    last_nn_layer: Identity\n")
            f.write("  input_dim: "+str(input_dim) + "\n")
            f.write("  latent_dim: " + str(latent_dim) + "\n")

            f.write("tracing:\n")
            f.write("  show_images: False\n")
            f.write("  images_to_show: 10\n")
            f.write("  visualize_latent: True\n")
            f.write("  batch_frequency: " + str(batch_frequency) + "\n")

            f.close()

if __name__ == "__main__":
    main()
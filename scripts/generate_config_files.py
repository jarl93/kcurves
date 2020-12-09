from helpers import create_dir
import numpy as np
import argparse
from constants import NUM_TESTS, DATA_SET, NUM_FUNCTIONS, FILLER, TESTS_PER_CASE
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

def write_config_training(path_config_files, list_names, list_train_paths, list_test_paths,
                          list_model_names, list_model_paths, list_evolution_paths):
    """
    Writes the config file for the training and testing.
    :param path_config_files:
    :param list_names:
    :param list_train_paths:
    :param list_test_paths:
    :param list_model_names:
    :param list_model_paths:
    :param list_evolution_paths:
    :return:
    """
    # TODO: Consider to make lists for the hyperparameters

    # train:
    # also equal to number of classes
    num_classes = 2
    batch_size = 128
    num_epochs = 200

    NUM_CASES_FIX = 12

    alpha_list = NUM_CASES_FIX*[10]

    # beta_type = "fixed"
    # beta_fixed = 0.001
    # beta_type = "down"
    # beta_type = "up"
    # beta_type = "up_down"
    # beta_fixed = 0.001
    # beta_min = 0.0001
    # beta_max = 0.001

    beta_type_list = NUM_CASES_FIX*["fixed"]
    beta_min_list = NUM_CASES_FIX*[0.0001]
    beta_max_list = NUM_CASES_FIX*[0.001]
    beta_fixed_list = NUM_CASES_FIX*[0.001]


    gamma_list = NUM_CASES_FIX*[10]

    type_dist = "axes"
    # type_dist = "points"
    # type_dist = "angle"

    type_loss = "entropy"
    # type_loss = "dist"

    lambda_ = 0.001
    lr = 0.001
    batch_frequency_loss = 1
    epochs_frequency_evolution = 20
    save_evolution = True
    evolution = True
    # test:
    mode_forced = "test"
    batch_size = 128


    # model:
    layer_sizes_encoder = [2, 2]
    layer_sizes_decoder = [2, 2]
    input_dim = 2
    latent_dim = 2

    # tracing:
    x_interval = [-1, 1]
    y_interval = [-1, 1]
    delta_interval = 0.01
    levels_contour = 20
    batch_frequency = 10

    for i in range(1, NUM_TESTS + 1):
        idx = (i -1) // TESTS_PER_CASE
        beta_min = beta_min_list[idx]
        beta_max = beta_max_list[idx]
        beta_type = beta_type_list[idx]
        beta_fixed = beta_fixed_list[idx]
        gamma_ = gamma_list[idx]
        alpha_ = alpha_list[idx]
        path = path_config_files + list_names[i] + ".yaml"
        f = open(path, "w")
        # write the files
        f.write("name: " + list_names[i] + "\n")
        f.write("data:\n")
        f.write("  data_set: " + str(DATA_SET) + "\n")
        f.write("  train: " + str(list_train_paths[i]) + "\n")
        f.write("  test: " + str(list_test_paths[i]) + "\n")
        f.write("  num_classes: " + str(num_classes) + "\n")

        f.write("train:\n")
        f.write("  batch_size: " + str(batch_size) + "\n")
        f.write("  num_epochs: " + str(num_epochs) + "\n")
        f.write("  alpha: " + str(alpha_) + "\n")
        f.write("  beta_type: "+str(beta_type)+ "\n")
        f.write("  beta_min: " + str(beta_min) + "\n")
        f.write("  beta_max: " + str(beta_max) + "\n")
        f.write("  beta_fixed: "+ str(beta_fixed) + "\n")
        f.write("  gamma: " + str(gamma_) + "\n")
        f.write("  lambda: " + str(lambda_) + "\n")
        f.write("  type_loss: " + str(type_loss) + "\n")
        f.write("  type_dist: " + str(type_dist) + "\n")
        f.write("  lr: " + str(lr) + "\n")
        f.write("  batch_frequency_loss: " + str(batch_frequency_loss) + "\n")
        f.write("  evolution : " + str(evolution) + "\n")
        f.write("  epochs_frequency_evolution : " + str(epochs_frequency_evolution) + "\n")
        f.write("  save_evolution : " + str(save_evolution) + "\n")

        f.write("test:\n")
        f.write("  mode_forced: " + str(mode_forced) + "\n")
        f.write("  batch_size: " + str(batch_size) + "\n")

        f.write("model:\n")
        f.write("  path: " + list_model_paths[i] + "\n")
        f.write("  evolution_path: " + list_evolution_paths[i]+ "\n")
        f.write("  name: " + list_model_names[i] + "\n")
        f.write("  save: True\n")
        f.write("  device: cpu\n")
        f.write("  encoder:\n")
        f.write("    layer_sizes: " + str(layer_sizes_encoder) + "\n")
        f.write("    last_nn_layer: Identity\n")
        f.write("  decoder:\n")
        f.write("    layer_sizes: " + str(layer_sizes_decoder) + "\n")
        f.write("    last_nn_layer: Identity\n")
        f.write("  input_dim: " + str(input_dim) + "\n")
        f.write("  latent_dim: " + str(latent_dim) + "\n")

        f.write("tracing:\n")
        f.write("  show_images: False\n")
        f.write("  images_to_show: 10\n")
        f.write("  visualize_latent: True\n")
        f.write("  x_interval: " + str(x_interval) + "\n")
        f.write("  y_interval: " + str(y_interval) + "\n")
        f.write("  delta_interval: " + str(delta_interval) + "\n")
        f.write("  levels_contour: " + str(levels_contour) + "\n")
        f.write("  batch_frequency: " + str(batch_frequency) + "\n")


        f.close()

    return None


def write_config_synthetic_clusters_generation(path_config_generation_files, list_names, list_train_paths,
                                                list_test_paths, list_plot_paths):
    """
    Writes config generation files for the synthetic clusters data set.
    :param path_config_generation_files:
    :param list_names:
    :param list_train_paths:
    :param list_test_paths:
    :param list_plot_paths:
    :return:
    """
    # TODO: Consider to make lists for the hyperparameters
    # TODO: Consider to make variables for the boolean hyperparameters
    # also equal to number of classes
    num_centers = 2
    dim = 2
    center_box = [-100, 100]
    cluster_std = 5
    scale_factor = 10
    num_samples_train = 5000
    num_samples_test = 1000
    for i in range(1, NUM_TESTS + 1):
        random_state = (i -1) % TESTS_PER_CASE
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

    return None


def write_config_synthetic_functions_generation(path_config_generation_files, list_names, list_train_paths,
                                                list_test_paths, list_plot_paths):
    """
    Writes config generation files for the synthetic functions data set.
    :param path_config_generation_files:
    :param list_names:
    :param list_train_paths:
    :param list_test_paths:
    :param list_plot_paths:
    :return:
    """
    # TODO: Consider to make lists for the hyperparameters
    # TODO: Consider to make variables for the boolean hyperparameters
    # list of hyperparameters for functions
    names_F = ["F1", "F2"]

    # paramters for case x < y
    # amp = [10, 10]
    # frec = [0.1, 0.1]
    # interval = [[0, 10], [0,10]]
    # # shift = [0, 30]
    # d_shift = 50

    # paramters for case linear separable
    amp = [3, 3]
    frec = [0.01, 0.01]
    interval = [[0, 100], [0, 100]]
    # shift = [0, 30]
    d_shift = 30

    char_to_plot = ['x', 'o']
    color_to_plot = ['red', 'green']
    # class distribution: 50-50
    train_num_samples = [3000, 3000]
    test_num_samples = [500, 500]

    # # class distribution: 70-30
    # train_num_samples = [4200, 1800]
    # test_num_samples = [700, 300]

    # # class distribution: 90-10
    # train_num_samples = [5400, 600]
    # test_num_samples = [900, 100]


    scale = True
    scale_factor = 10
    normalize = True
    non_linear = "sigmoid"
    list_dimensions = [[10, 2], [50, 10], [100, 50]]

    shift_list = []

    for i in range(1, NUM_TESTS + 1):
        path = path_config_generation_files + list_names[i] + ".yaml"
        f = open(path, "w")
        # write the file
        f.write("name: " + list_names[i] + "\n")

        # code to generate different cases
        if i <= TESTS_PER_CASE:
            # enforce a separation of at least d_shift units between the two functions
            shift_1 = np.random.randint(-50, 50)
            shift_2 = np.random.randint(-50, 50)
            while abs(shift_1 - shift_2) < d_shift:
                shift_1 = np.random.randint(-50, 50)
                shift_2 = np.random.randint(-50, 50)

            # fixed case
            # shift_1 = -40
            # shift_2 = 10

            shift = [shift_1, shift_2]
            shift_list.append(shift)
        else:
            idx = (i-1) % TESTS_PER_CASE
            shift = shift_list[idx]

        # functions
        for j in range(NUM_FUNCTIONS):
            f.write(names_F[j]+":\n")
            f.write("  amp: " + str(amp[j]) + "\n")
            f.write("  frec: " + str(frec[j]) + "\n")
            f.write("  interval: " + str(interval[j]) + "\n")
            f.write("  shift: " + str(shift[j]) + "\n")
            f.write("  char_to_plot: " + str(char_to_plot[j]) + "\n")
            f.write("  color_to_plot: " + str(color_to_plot[j]) + "\n")
            f.write("  train_num_samples: " + str(train_num_samples[j]) + "\n")
            f.write("  test_num_samples: " + str(test_num_samples[j]) + "\n")

        # data parameters for generation
        f.write("data:\n")
        f.write("  save: True\n")
        f.write("  plot: True\n")
        f.write("  scale: " + str(scale) + "\n")
        f.write("  scale_factor: " + str(scale_factor) + "\n")
        f.write("  normalize: " + str(normalize) + "\n")
        f.write("  transformation: False\n")
        f.write("  train:\n")
        f.write("    path: " + list_train_paths[i] + "\n")
        f.write("  test:\n")
        f.write("    path: " + list_test_paths[i] + "\n")
        f.write("  plots:\n")
        f.write("    path: " + list_plot_paths[i] + "\n")

        # define transformation
        f.write("transformation:\n")
        f.write("  non_linear: " + str(non_linear) + "\n")
        f.write("  list_dimensions:\n")
        for dimension in list_dimensions:
            f.write("    - "+str(dimension)+"\n")

        f.close()

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--all', action='store_true',
                    help='Generate config files for the generation process and for training (and testing).')

    ap.add_argument('-g', '--generation', action='store_true',
                    help='Generate config files for the generation process.')

    ap.add_argument('-t', '--training', action='store_true',
                    help='Generate config files for training and testing.')

    args = ap.parse_args()

    # create directory for the config files
    path_config_files = "./configs/synthetic_"+FILLER+"/"
    create_dir(path_config_files)

    # main paths for training data, test data and plots (generated data)
    train_path = "./data/synthetic_"+FILLER+"/train/"
    test_path = "./data/synthetic_"+FILLER+"/test/"
    plot_path = "./data/synthetic_"+FILLER+"/plots/"

    # code to generate config files that generate data
    if args.generation or args.all:
        # create directory for the config generation files

        path_config_generation_files = "./configs/synthetic_"+FILLER+"_generation/"
        create_dir(path_config_generation_files)
        name = "synthetic_"+FILLER+"_generation"
        # lists with the paths for the training data, test data and plots
        list_names = generate_list(name, NUM_TESTS, "_")
        list_train_paths = generate_list(train_path, NUM_TESTS, "/")
        list_test_paths = generate_list(test_path, NUM_TESTS, "/")
        list_plot_paths = generate_list(plot_path, NUM_TESTS, "/")

        # write config file for generation
        if DATA_SET == "synthetic_functions":
            write_config_synthetic_functions_generation(path_config_generation_files, list_names, list_train_paths,
                                                        list_test_paths, list_plot_paths)
        elif DATA_SET == "synthetic clusters":
            write_config_synthetic_clusters_generation(path_config_generation_files, list_names, list_train_paths,
                                                       list_test_paths, list_plot_paths)


    # code to generate config files for training and testing
    if args.training or args.all:

        # create directories for config files and define path
        path_config_files = "./configs/synthetic_"+FILLER+"/"
        create_dir(path_config_files)
        name = "synthetic_"+FILLER
        model_name = "model_"+FILLER
        model_path = "./models/synthetic_"+FILLER+"/"
        evolution_path = "./models/synthetic_"+FILLER+"_evolution/"

        # create lists with train, test and model paths
        list_names = generate_list(name, NUM_TESTS, "_")
        list_train_paths = generate_list(train_path, NUM_TESTS, "/")
        list_test_paths = generate_list(test_path, NUM_TESTS, "/")
        list_model_names = generate_list(model_name, NUM_TESTS, "_")
        list_model_paths = generate_list(model_path, NUM_TESTS, "/")
        list_evolution_paths = generate_list(evolution_path, NUM_TESTS, "/")

        # write config file for training
        write_config_training(path_config_files, list_names, list_train_paths, list_test_paths,
                              list_model_names, list_model_paths, list_evolution_paths)


if __name__ == "__main__":
    main()
from generate_config_files import generate_list
from helpers import create_dir
from constants import NUM_TESTS, FILLER
if __name__ == "__main__":


    # create directory for data
    create_dir("./data/synthetic_"+FILLER)


    # define paths for training and test data, plots ad models
    train_path = "./data/synthetic_"+FILLER+"/train/"
    test_path = "./data/synthetic_"+FILLER+"/test/"
    plot_path = "./data/synthetic_"+FILLER+"/plots/"
    model_path = "./models/synthetic_"+FILLER+"/"
    evolution_path = "./models/synthetic_" + FILLER + "_evolution/"

    # create directories for models and graphs
    create_dir("./models/synthetic_" + FILLER)
    create_dir("./models/synthetic_" + FILLER + "_evolution")


    # generate lists of paths
    list_train_paths = generate_list(train_path, NUM_TESTS, "/")
    list_test_paths = generate_list(test_path, NUM_TESTS, "/")
    list_plot_paths = generate_list(plot_path, NUM_TESTS, "/")
    list_model_paths = generate_list(model_path, NUM_TESTS, "/")
    list_evolution_paths = generate_list(evolution_path, NUM_TESTS, "/")


    # create directories
    for i in range(1, NUM_TESTS+1):
        create_dir(list_train_paths[i])
        create_dir(list_test_paths[i])
        create_dir(list_plot_paths[i])
        create_dir(list_model_paths[i])
        create_dir(list_evolution_paths[i])
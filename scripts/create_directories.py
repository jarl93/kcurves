from generate_config_files import generate_list
from helpers import create_dir
if __name__ == "__main__":

    NUM_TESTS = 20
    train_path = "./data/synthetic_clusters/train/"
    test_path = "./data/synthetic_clusters/test/"
    plot_path = "./data/synthetic_clusters/plots/"
    model_path = "./models/synthetic_clusters/"

    list_train_paths = generate_list(train_path, NUM_TESTS, "/")
    list_test_paths = generate_list(test_path, NUM_TESTS, "/")
    list_plot_paths = generate_list(plot_path, NUM_TESTS, "/")
    list_model_paths = generate_list(model_path, NUM_TESTS, "/")

    for i in range(1, NUM_TESTS+1):
        create_dir(list_train_paths[i])
        create_dir(list_test_paths[i])
        create_dir(list_plot_paths[i])
        create_dir(list_model_paths[i])
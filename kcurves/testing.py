# libraries
import torch
import numpy as np
from helpers import load_config
from helpers import imshow
from helpers import plot_X2D_visualization, create_writer, get_regularization_hyperparameters, make_string_from_dic
from torch.utils.data import DataLoader
from _datetime import datetime
from clustering import k_means

def test(cfg_path, model, data_set):
    """
    Add documentation.
    :param cfg_path:
    :param model:
    :param data_set:
    :return:
    """

    cfg_file = load_config(cfg_path)

    device = cfg_file["model"]["device"]
    mode_forced = cfg_file["test"]["mode_forced"]
    batch_size = cfg_file["test"]["batch_size"]
    visualize_latent = cfg_file["tracing"]["visualize_latent"]
    num_classes = cfg_file["data"]["num_classes"]
    show_images = cfg_file["tracing"]["show_images"]
    images_to_show = cfg_file["tracing"]["images_to_show"]

    # get the regularization hyperparameters
    dic_regularization_types, dic_scalar_hyperparameters = get_regularization_hyperparameters(cfg_path)
    str_reg_types = make_string_from_dic(dic_regularization_types)
    str_scalar_hyperparameters = make_string_from_dic(dic_scalar_hyperparameters)

    # create a path for the log directory that includes the date and the hyperparameters for the regularization
    path_log_dir = cfg_file["model"]["path"] + "log_testing_" + datetime.now().strftime("%d.%m.%Y-%H:%M:%S") + \
                   "\n" + str_reg_types + "\n" + str_scalar_hyperparameters

    writer = create_writer(path_log_dir)

    # load the model from training
    path = cfg_file["model"]["path"] + cfg_file["model"]["name"]
    model.load_state_dict(torch.load(path))
    model.eval()

    if mode_forced == 'test':
        _, test_dataset = data_set
    else: # this in case we want to test with the training data
        test_dataset, _ = data_set

    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

    print("Starting testing on {} data in mode {}...".format(cfg_file["data"]["data_set"], mode_forced))

    # numpy array to store the vectors in the latent space for each sample
    X_2D = None
    labels = None
    list_images = [] # list for images in case of MNIST data set

    for batch_idx, data in enumerate(test_loader):

        # get the data and labels from the generator
        x, y = data

        # store a random image from the batch
        idx_random = np.random.randint(0, len(y))
        if cfg_file["data"]["data_set"] == "mnist":
            # save the original image for later comparison
            img_original = x[idx_random].squeeze()

        # print("x shape: ", x.shape)
        # print("y shape: ", y.shape)

        # Send the data to the device
        x = x.to(device)

        # Resize the input accordingly
        x = x.view(-1, model.input_dim)

        # Encode the data to see how the result looks
        h = model.encoder(x).detach().numpy()

        if batch_idx == 0:
            X_2D = h
            labels = y
        else:
            X_2D = np.vstack((X_2D, h))
            labels = np.hstack((labels, y))

        # Get the reconstrunction from the autoencoder
        x_reconstructed = model(x)

        if cfg_file["data"]["data_set"] == "mnist":
            if show_images and batch_idx < images_to_show:
                # resize the tensor to see the image
                img_reconstructed = x_reconstructed.view(-1, 28, 28).detach().numpy()[idx_random]
                list_images.append((img_original, img_reconstructed))

    print("X_2D shape: ", X_2D.shape)

    if show_images:
        writer.add_figure('originals vs reconstructed', imshow(list_images))
        #imshow(list_images) # show images in case of MNIST data set

    if visualize_latent:
        title = "Encoder output on {} data in mode {} with true labels".format(cfg_file["data"]["data_set"], mode_forced)

        writer.add_figure('01 Visualization of {} data latent space 2D'.format(mode_forced),
                          plot_X2D_visualization(X_2D, labels, title = title, num_classes = num_classes))

        # TODO: Consider to put the following code in a different module
        # code for visualization of clustering in latent space
        centers_k_means, labels_k_means = k_means(X_2D, n_clusters = num_classes)
        title = "K-means on encoder output on {} data in mode {}".format(cfg_file["data"]["data_set"],
                                                                                          mode_forced)

        writer.add_figure('02 Visualization of k-means on {} data in latent space 2D'.format(mode_forced),
                          plot_X2D_visualization(X_2D, labels_k_means, title=title, num_classes=num_classes,
                                                 cluster_centers=centers_k_means))

        # leave the repeated code, because otherwise the last added image does not appear.
        # It seems to be a bug from tensorboard, although more investigation is required.
        writer.add_figure('02 Visualization of k-means on {} data in latent space 2D'.format(mode_forced),
                          plot_X2D_visualization(X_2D, labels_k_means, title=title, num_classes=num_classes,
                                                 cluster_centers=centers_k_means))


    print("Testing DONE!")

    return None

# libraries
import torch
import torch.nn as nn
import numpy as np
from helpers import plot_X2D_visualization, create_writer, get_hyperparameters, \
    make_string_from_dic, get_softmax, imshow,load_config, plot_2D_visualization_clusters, \
    softmax_pair, entropy_pair, distance_axes_pair
from torch.utils.data import DataLoader
from _datetime import datetime
from clustering import k_means


def test(cfg_path, model, data_set, mode_forced, mode, lap = "0"):
    """
    Add documentation.
    :param cfg_path:
    :param model:
    :param data_set:
    :return:
    """

    cfg_file = load_config(cfg_path)

    device = cfg_file["model"]["device"]
    batch_size = cfg_file["test"]["batch_size"]
    visualize_latent = cfg_file["tracing"]["visualize_latent"]
    a_x, b_x = cfg_file["tracing"]["x_interval"]
    a_y, b_y = cfg_file["tracing"]["y_interval"]
    delta_interval = cfg_file["tracing"]["delta_interval"]
    levels_contour = cfg_file["tracing"]["levels_contour"]
    num_classes = cfg_file["data"]["num_classes"]
    show_images = cfg_file["tracing"]["show_images"]
    images_to_show = cfg_file["tracing"]["images_to_show"]
    p_ref_opt = cfg_file["train"]["p_ref"]
    dist_classes = cfg_file["data"]["dist_classes"]

   # get the hyperparameters of the config file
    dic_hyperparameters = get_hyperparameters(cfg_path)
    str_hyperparameters = make_string_from_dic(dic_hyperparameters)

    # create a path for the log directory that includes the date and the hyperparameters
    if mode == "evolution":
        path_log_dir = cfg_file["model"]["evolution_path"] + "log_evolution_lap_" + str(lap) + "_" \
                       + datetime.now().strftime("%d.%m.%Y-%H:%M:%S") + "\n" + str_hyperparameters
    elif mode == "final":
        path_log_dir = cfg_file["model"]["path"] + "log_test_mode_" + str(mode_forced) \
                       + "\n" + str_hyperparameters

    writer = create_writer(path_log_dir)

    if mode == "final":
        # load the model from training in case the training is done
        path = cfg_file["model"]["path"] + cfg_file["model"]["name"]
        model.load_state_dict(torch.load(path))

    model.eval()

    train_dataset, test_dataset = data_set

    if mode_forced == 'test' and mode == 'final':
        test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    elif mode_forced == 'train' or mode == 'evolution':
        test_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = False)

    if mode == "evolution":
        print("Starting lap {} for evolution on {} data in mode {}...".format(lap, cfg_file["data"]["data_set"],
                                                                                mode_forced))
    elif mode == "final":
        print("Starting testing on {} data in mode {}...".format(cfg_file["data"]["data_set"], mode_forced))

    # numpy array to store the vectors in the latent space for each sample
    H_2D = None
    H1_2D = None
    H2_2D = None
    X_input = None
    X_reconstructed = None
    labels = None
    list_images = [] # list for images in case of MNIST data set

    for batch_idx, data in enumerate(test_loader):

        # get the data and labels from the generator
        x, y = data

        if cfg_file["data"]["data_set"] == "mnist":
            # store a random image from the batch
            idx_random = np.random.randint(0, len(y))
            # save the original image for later comparison
            img_original = x[idx_random].squeeze()

        # Send the data to the device
        x = x.to(device)

        # Resize the input accordingly
        x = x.view(-1, model.input_dim)

        x_numpy = x.detach().numpy()

        # Encode the data to see how the result looks
        #h, _, h1, h2 = model.encoder(x)

        h = model.encoder(x)
        h1 = h
        h2 = h

        # Get the reconstruction from the autoencoder

        x_reconstructed = model.decoder(h)

        x_reconstructed_numpy = x_reconstructed.detach().numpy()

        h_numpy = h.detach().numpy()
        h1_numpy = h1.detach().numpy()
        h2_numpy = h2.detach().numpy()

        if batch_idx == 0:
            X_input = x_numpy
            X_reconstructed = x_reconstructed_numpy
            H_2D = h_numpy
            H1_2D = h1_numpy
            H2_2D = h2_numpy
            labels = y
        else:
            X_input = np.vstack((X_input, x_numpy))
            X_reconstructed = np.vstack((X_reconstructed, x_reconstructed_numpy))
            H_2D = np.vstack((H_2D, h_numpy))
            H1_2D = np.vstack((H1_2D, h1_numpy))
            H2_2D = np.vstack((H2_2D, h2_numpy))
            labels = np.hstack((labels, y))


        if cfg_file["data"]["data_set"] == "mnist":
            if show_images and batch_idx < images_to_show:
                # resize the tensor to see the image
                img_reconstructed = x_reconstructed.view(-1, 28, 28).detach().numpy()[idx_random]
                list_images.append((img_original, img_reconstructed))

    # get the softmax to the distance to the axes
    H_2D_softmax = get_softmax(np.absolute(H_2D))

    # compute the accuracy
    prediction = np.argmax(H_2D_softmax, axis = 1)
    accuracy_1 = np.sum( prediction == labels)/len(labels)
    accuracy_2 = np.sum( prediction != labels)/len(labels)
    accuracy = ["accuracy_1 = "+str(accuracy_1), "accuracy_2 = " +str(accuracy_2)]


    if show_images:
        writer.add_figure('originals vs reconstructed', imshow(list_images))
        #imshow(list_images) # show images in case of MNIST data set

    if visualize_latent:

        # compute the entropy of softmax([a_x, b_x] x [a_y, b_y])
        x_interval = np.arange(a_x, b_x, delta_interval)
        y_interval = np.arange(a_y, b_y, delta_interval)
        x_i, y_i = np.meshgrid(x_interval, y_interval)
        z_i_ent = entropy_pair(*softmax_pair(x_i, y_i))
        z_i_dist = distance_axes_pair(x_i, y_i)
        list_vars = [x_i, y_i, z_i_ent, z_i_dist]

        # make the list for the outputs (inputs) of the auto-encoder
        list_X = [X_input, X_reconstructed, H_2D, H_2D_softmax, H1_2D, H2_2D]

        # titles = ["Input", "Reconstruction", "Latent Space", "Softmax Latent Space",
        #           "Objective entropy in [{}, {}] x [{}, {}]".format(a_x, b_x, a_y, b_y),
        #           "Objective distance in [{}, {}] x [{}, {}]".format(a_x, b_x, a_y, b_y)]

        titles = ["Input(" + dist_classes + ")", "Reconstruction", "Latent Space",
                  "Softmax LS, p_ref = " + str(p_ref_opt), "Transformation 1", "Transformation 2"]

        writer.add_figure('01 Visualization of Encoder outputs (or inputs)',
                          plot_2D_visualization_clusters(list_X = list_X, list_vars = list_vars , labels = labels,
                                                         titles = titles, num_classes=num_classes,
                                                         levels_contour = levels_contour, accuracy = accuracy))

        writer.add_figure('01 Visualization of Encoder outputs (or inputs)',
                          plot_2D_visualization_clusters(list_X=list_X, list_vars = list_vars, labels=labels,
                                                         titles=titles, num_classes=num_classes,
                                                         levels_contour=levels_contour, accuracy = accuracy))

        # TODO: Consider to put the following code in a different module
        # code for visualization of clustering in latent space
        # centers_k_means, labels_k_means = k_means(H_2D, n_clusters = num_classes)
        # title = "K-means on encoder output on {} data in mode {}".format(cfg_file["data"]["data_set"],
        #                                                                                   mode_forced)
        #
        # writer.add_figure('02 Visualization of k-means on {} data in latent space 2D'.format(mode_forced),
        #                   plot_X2D_visualization(H_2D, labels_k_means, title=title, num_classes=num_classes,
        #                                          cluster_centers=centers_k_means))
        #
        # leave the repeated code, because otherwise the last added image does not appear.
        # It seems to be a bug from tensorboard, although more investigation is required.
        # writer.add_figure('02 Visualization of k-means on {} data in latent space 2D'.format(mode_forced),
        #                   plot_X2D_visualization(H_2D, labels_k_means, title=title, num_classes=num_classes,
        #                                          cluster_centers=centers_k_means))

    if mode == "evolution":
        print("Lap {} for evolution DONE!".format(lap))
    elif mode == "final":
        print("Final testing DONE!")

    return None

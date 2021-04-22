# libraries
import torch
import torch.nn as nn
import numpy as np
from helpers import plot_X2D_visualization, create_writer, get_hyperparameters, \
    make_string_from_dic, get_softmax, imshow,load_config, plot_2D_visualization_clusters, \
    softmax_pair, entropy_pair, distance_axes_pair, get_interpolation, get_purity, get_ARI, get_NMI
from torch.utils.data import DataLoader
from _datetime import datetime
from helpers import pairwise_distances, pairwise_distances_segments
from clustering import k_means
from constants import DEVICE
from initialization import PCA_proporcional_dist

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
    #visualization = cfg_file["tracing"]["visualization"]
    a_x, b_x = cfg_file["tracing"]["x_interval"]
    a_y, b_y = cfg_file["tracing"]["y_interval"]
    delta_interval = cfg_file["tracing"]["delta_interval"]
    levels_contour = cfg_file["tracing"]["levels_contour"]
    num_classes = cfg_file["data"]["num_classes"]
    percentage_K = cfg_file["train"]["percentage_K"]
    show_images = cfg_file["tracing"]["show_images"]
    images_to_show = cfg_file["tracing"]["images_to_show"]
    #p_ref_opt = cfg_file["train"]["p_ref"]
    type_dist = cfg_file["train"]["type_dist"]
    input_dim = cfg_file["model"]["input_dim"]
    latent_dim = cfg_file["model"]["latent_dim"]
    num_points_inter = cfg_file["tracing"]["num_points_inter"]

   # get the hyperparameters of the config file
    dic_hyperparameters = get_hyperparameters(cfg_path)
    str_hyperparameters = make_string_from_dic(dic_hyperparameters)

    # create a path for the log directory that includes the date and the hyperparameters
    if mode == "evolution":
        path_log_dir = cfg_file["model"]["evolution_path"] + "log_evolution_lap_" + str(lap) + "_" \
                       + datetime.now().strftime("%d.%m.%Y-%H:%M:%S") + "\n" + str_hyperparameters
    elif mode == "final":
        path_log_dir = cfg_file["model"]["path"] + "log_test_mode_" + str(mode_forced) \
                       + datetime.now().strftime("%d.%m.%Y-%H:%M:%S") + "\n" + str_hyperparameters

    writer = create_writer(path_log_dir)


    # load the model depending on evolution or final mode
    if mode == "evolution":
        path = cfg_file["model"]["evolution_path"] + cfg_file["model"]["name"] + "_lap_" + str(lap)
    else:
        path = cfg_file["model"]["path"] + cfg_file["model"]["name"]

    model.load_state_dict(torch.load(path))
    model.eval()

    train_dataset, test_dataset = data_set

    if mode_forced == 'test' and mode == 'final':
        test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
        centers_true = np.load(cfg_file["data"]["test"] + "centers_test.npy")
    elif mode_forced == 'train' or mode == 'evolution':
        test_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = False)
        centers_true = np.load(cfg_file["data"]["train"] + "centers_train.npy")

    if mode == "evolution":
        print("Starting lap {} for evolution on {} data in mode {}...".format(lap, cfg_file["data"]["data_set"],
                                                                                mode_forced))
    elif mode == "final":
        print("Starting testing on {} data in mode {}...".format(cfg_file["data"]["data_set"], mode_forced))

    # numpy array to store the vectors in the latent space for each sample
    H_latent = None
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

        x_numpy = x.cpu().detach().numpy()

        h = model.encoder(x)

        # Get the reconstruction from the autoencoder
        x_reconstructed = model.decoder(h)
        x_reconstructed_numpy = x_reconstructed.cpu().cpu().detach().numpy()
        h_numpy = h.cpu().detach().numpy()

        if batch_idx == 0:
            X_input = x_numpy
            X_reconstructed = x_reconstructed_numpy
            H_latent = h_numpy
            labels = y
        else:
            X_input = np.vstack((X_input, x_numpy))
            X_reconstructed = np.vstack((X_reconstructed, x_reconstructed_numpy))
            H_latent = np.vstack((H_latent, h_numpy))
            labels = np.hstack((labels, y))

        if cfg_file["data"]["data_set"] == "mnist":
            if show_images and batch_idx < images_to_show:
                # resize the tensor to see the image
                img_reconstructed = x_reconstructed.view(-1, 28, 28).cpu().detach().numpy()[idx_random]
                list_images.append((img_original, img_reconstructed))

    H_latent_tensor = torch.from_numpy(H_latent).to(DEVICE)


    # run k-means on latent space
    centers_k_means, predicitons_kmeans = k_means(X=H_latent, centers_init='k-means++', n_clusters=num_classes)

    # run k-means on latent space
    if type_dist == "points":
        centers_latent = centers_k_means
        np.save(cfg_file["data"]["train"] + "rep_latent", centers_latent)
    elif type_dist == "segments":
        rep_latent = PCA_proporcional_dist(H_latent, num_classes, percentage_K)
        rep_latent = rep_latent.cpu().detach().numpy()
        np.save(cfg_file["data"]["train"] + "rep_latent", rep_latent)


    if type_dist == "points":
        centers_rec = model.decoder(model.rep)
        centers_rec_numpy = centers_rec.cpu().detach().numpy()
        # compute the distances to the learned representatives
        rep = model.rep
        centers_latent = rep.cpu().detach().numpy()
        dist = pairwise_distances(H_latent_tensor, rep)
        list_rep = [centers_true, centers_rec_numpy, centers_latent, centers_k_means]
        list_inter = [None, None, None, None]
    elif type_dist == "segments":
        s1 = model.rep[:,:latent_dim]
        s2 = model.rep[:,latent_dim:]
        s_inter = get_interpolation(s1, s2, num_points_inter)
        s_inter_latent = s_inter.cpu().detach().numpy()
        s_latent = model.rep.cpu().detach().numpy()

        print("s1 shape: ", s1.shape)
        print("s2 shape: ", s2.shape)

        s1_rec = model.decoder(s1)
        s2_rec = model.decoder(s2)
        s_inter_rec = model.decoder(s_inter)
        s_inter_rec_np = s_inter_rec.cpu().detach().numpy()

        print("shape s_inter_rec_np = ", s_inter_rec_np.shape)


        s_rec = torch.cat((s1_rec, s2_rec), 1)
        s_rec_numpy = s_rec.cpu().detach().numpy()
        #print("s_rec_numpy: ", s_rec_numpy)
        dist = pairwise_distances_segments(H_latent_tensor, model.rep)
        list_rep = [centers_true, s_rec_numpy, s_latent, centers_k_means]
        list_inter = [None, s_inter_rec_np, s_inter_latent, None]


    dist_numpy = dist.cpu().detach().numpy()


    # compute the accuracy
    predictions = np.argmin(dist_numpy, axis = 1)

    # metrics for the our clustering algorithm
    purity = get_purity(labels, predictions)
    NMI = get_NMI(labels, predictions)
    ARI = get_ARI(labels, predictions)
    metrics = ["ACC = " + str(purity), "NMI = " + str(NMI), "ARI = " + str(ARI)]



    # metrics for vanilla k-means
    purity_kmeans = get_purity(labels, predicitons_kmeans)
    NMI_kmeans = get_NMI(labels, predicitons_kmeans)
    ARI_kmeans = get_ARI(labels, predicitons_kmeans)
    metrics_kmeans = ["ACC = " + str(purity_kmeans), "NMI = " + str(NMI_kmeans), "ARI = " + str(ARI_kmeans)]




    if show_images:
        writer.add_figure('originals vs reconstructed', imshow(list_images))
        #imshow(list_images) # show images in case of MNIST data set

    if visualize_latent:

        # ------------------------- Frozen code ------------------- #
        # compute the entropy of softmax([a_x, b_x] x [a_y, b_y])
        # x_interval = np.arange(a_x, b_x, delta_interval)
        # y_interval = np.arange(a_y, b_y, delta_interval)
        # x_i, y_i = np.meshgrid(x_interval, y_interval)
        # z_i_ent = entropy_pair(*softmax_pair(x_i, y_i))
        # z_i_dist = distance_axes_pair(x_i, y_i)
        # list_vars = [x_i, y_i, z_i_ent, z_i_dist]


        # make the list for the outputs (inputs) of the auto-encoder
        list_X = [X_input, X_reconstructed, H_latent, H_latent]

        titles = ["Input", "Reconstruction", "Latent Space", "k-means on Latent Space "
                                                             ""]

        writer.add_figure('01 Visualization of Encoder outputs (or inputs)',
                          plot_2D_visualization_clusters(list_X = list_X, labels = labels,
                                                         predictions_kmeans = predicitons_kmeans, predictions = predictions,
                                                         list_rep = list_rep, list_inter = list_inter,
                                                         titles=titles, num_classes=num_classes,
                                                         metrics=metrics, metrics_kmeans= metrics_kmeans,
                                                         type_dist=type_dist))


        # TODO: Consider to put the following code in a different module
        # code for visualization of clustering in latent space
        # centers_k_means, labels_k_means = k_means(H_latent, n_clusters = num_classes)
        # title = "K-means on encoder output on {} data in mode {}".format(cfg_file["data"]["data_set"],
        #                                                                                   mode_forced)
        #
        # writer.add_figure('02 Visualization of k-means on {} data in latent space 2D'.format(mode_forced),
        #                   plot_X2D_visualization(H_latent, labels_k_means, title=title, num_classes=num_classes,
        #                                          cluster_centers=centers_k_means))
        #
        # leave the repeated code, because otherwise the last added image does not appear.
        # It seems to be a bug from tensorboard, although more investigation is required.
        # writer.add_figure('02 Visualization of k-means on {} data in latent space 2D'.format(mode_forced),
        #                   plot_X2D_visualization(H_latent, labels_k_means, title=title, num_classes=num_classes,
        #                                          cluster_centers=centers_k_means))


    if mode == "evolution":
        print(metrics)
        print("Lap {} for evolution DONE!".format(lap))

    elif mode == "final":
        print (metrics)
        print("Final testing DONE!")
        writer.close()

    return purity, NMI, ARI

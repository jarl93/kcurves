# libraries
from helpers import load_config, pairwise_distances_segments
from model import AE
import numpy as np
import torch
from sklearn.decomposition import PCA
from constants import DEVICE
from helpers import get_hidden_layers
def init_model(cfg_path, verbose=True):
    """
    Add documentation
    :param encoder_layer_sizes:
    :param decoder_layer_sizes:
    :param input_dim:
    :param latent_dim:
    :param verbose:
    :return:
    """
    cfg_file = load_config(cfg_path)
    encoder_layer_sizes = cfg_file["model"]["encoder"]["layer_sizes"]
    decoder_layer_sizes = cfg_file["model"]["decoder"]["layer_sizes"]
    input_dim = cfg_file["model"]["input_dim"]
    latent_dim = cfg_file["model"]["latent_dim"]
    # last_nn_layer_encoder = cfg_file["model"]["encoder"]["last_nn_layer"]
    # last_nn_layer_decoder = cfg_file["model"]["decoder"]["last_nn_layer"]
    device = cfg_file["model"]["device"]
    K = cfg_file["data"]["num_classes"]
    centers_init_type = cfg_file["train"]["centers_init_type"]
    min_init = cfg_file["train"]["min_init"]
    diff_init = cfg_file["train"]["diff_init"]
    percentage_K = cfg_file["train"]["percentage_K"]
    type_dist = cfg_file["train"]["type_dist"]
    X_train = np.load(cfg_file["data"]["train"] + "X_train.npy")
    if verbose:
        print("Initialization of the model...")


    if type_dist == "points":
        if centers_init_type == "forgy": # forgy initialization
            rep_init = forgy_initialization(X_train, K)
        elif centers_init_type == "random": # random initialization
            rep_init = min_init + diff_init * np.random.rand(K, latent_dim)
        np.save(cfg_file["data"]["train"] + "rep_init", rep_init)
        np.save(cfg_file["data"]["train"] + "centers_init", rep_init)
    elif type_dist == "segments":
        if centers_init_type == "knearest_kdivision":
            rep_init = knearest_kdivision(X_train, K, percentage_K)
        elif centers_init_type == "PCA_kdivision":
            rep_init = PCA_kdivision(X_train, K, percentage_K)
        elif centers_init_type == "knearest_furthest":
            rep_init = knearest_furthest(X_train, K, percentage_K)
        elif centers_init_type == "PCA_furthest":
            rep_init = PCA_furthest(X_train, K, percentage_K)
        elif centers_init_type == "PCA_proportional_dist":
            rep_init = PCA_proporcional_dist(X_train, K, percentage_K)

        elif centers_init_type == "max_length_random":
            max_length_start = cfg_file["train"]["max_length_start"]
            s1 = min_init + diff_init * np.random.rand(K, latent_dim)
            s2 = s1 + max_length_start * np.random.rand(K, latent_dim)
            rep_init = np.hstack((s1, s2))
        elif centers_init_type == "random":
            rep_init = np.random.rand(K, 2 * latent_dim)
        if type(rep_init) == torch.Tensor:
            rep_init = rep_init.cpu().detach().numpy()
        np.save(cfg_file["data"]["train"] + "rep_init", rep_init)
        np.save(cfg_file["data"]["train"] + "centers_init", rep_init[:,:input_dim])


    print("rep_init shape: ", rep_init.shape)
    # Define the model as an autoencoder
    model = AE(input_dim=input_dim, encoder_layer_sizes = encoder_layer_sizes,
               decoder_layer_sizes = decoder_layer_sizes, latent_dim = latent_dim,
               rep_init = rep_init, rep_type = type_dist)

    model = xavier_initialization(model)

    model = model.to(device)


    if verbose:
        print("Model: ", model)

    return model

def xavier_initialization(model):
    layers = get_hidden_layers(model)
    for layer in layers:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    return model

def forgy_initialization(X, K):
    N = X.shape[0]
    idx = np.random.choice(N, K)
    centers_init = X[idx]
    return centers_init

def knearest_kdivision(X, K, percentage_K):

    N = X.shape[0]
    idx = np.random.choice(N)
    length_k = N // K
    scaled_length_k = int(percentage_K * length_k)
    s1 = X[idx]
    distance_s1 = np.sum((X - s1) ** 2, axis=1)
    neigh = np.argsort(distance_s1)
    idx = np.random.choice(scaled_length_k - 1) + 1
    s2 = X[neigh[idx]] # choose randomly among the closest ones
    s = np.hstack((s1, s2))

    for i in range(1, K):
        s1_range_i = neigh[i*length_k: (i+1)*length_k]
        idx_i = np.random.choice(length_k)
        s1_i = X[s1_range_i[idx_i]]
        distance_s1_i = np.sum((X - s1_i) ** 2, axis=1)
        neigh_i = np.argsort(distance_s1_i)
        idx_i = np.random.choice(scaled_length_k - 1) + 1
        s2_i = X[neigh_i[idx_i]] # choose randomly among the closest ones
        s12_i = np.hstack((s1_i, s2_i))
        s = np.vstack((s, s12_i))

    return s


def knearest_furthest(X, K, percentage_K):
    N = X.shape[0]
    length_k = N // K
    scaled_length_k = int(percentage_K * length_k)

    for i in range(K):
        if i == 0:
            idx = np.random.choice(N)
            s1_i = X[idx]
            distance_s1_i = np.sum((X - s1_i) ** 2, axis=1)
            neigh_i = np.argsort(distance_s1_i)
            idx_i = np.random.choice(scaled_length_k - 1) + 1
            s2_i = X[neigh_i[idx_i]]  # choose randomly among the closest ones
            s = np.hstack((s1_i, s2_i))
        else:
            s1_range_i = neigh_i[-length_k:]
            idx_i = np.random.choice(length_k)
            s1_i = X[s1_range_i[idx_i]]
            X = X[neigh_i[length_k:]]
            distance_s1_i = np.sum((X - s1_i) ** 2, axis=1)
            neigh_i = np.argsort(distance_s1_i)
            idx_i = np.random.choice(scaled_length_k - 1) + 1
            s2_i = X[neigh_i[idx_i]]  # choose randomly among the closest ones
            s12_i = np.hstack((s1_i, s2_i))
            s = np.vstack((s, s12_i))

    return s

def get_s2_PCA(neigh, s1):
    if torch.isnan(neigh).any():
        raise ValueError(" neigh has  nan values!")

    neigh = neigh.cpu().detach().numpy()
    s1 = s1.cpu().detach().numpy()
    pca = PCA(n_components=2)
    pca.fit(neigh)
    c_std = 2.0 + np.random.rand()
    length_s = c_std * np.sqrt(pca.explained_variance_[0])
    v_dir = pca.components_[0]
    s2_pos = s1 + length_s * v_dir
    s2_neg = s1 - length_s * v_dir

    # check which direction is more convenient
    s_pos = torch.from_numpy(np.hstack((s1, s2_pos)).reshape(1, -1)).to(DEVICE)
    s_neg = torch.from_numpy(np.hstack((s1, s2_neg)).reshape(1, -1)).to(DEVICE)
    neigh_torch = torch.from_numpy(neigh).to(DEVICE)
    d_pos = pairwise_distances_segments(neigh_torch, s_pos)
    d_neg = pairwise_distances_segments(neigh_torch, s_neg)
    if torch.sum(d_pos) < torch.sum (d_neg):
        s2_numpy = s2_pos
    else:
        s2_numpy = s2_neg

    s2 = torch.from_numpy(s2_numpy).to(DEVICE)
    return s2

def PCA_proporcional_dist(X, K, percentage_K):

    N = X.shape[0]
    X = torch.from_numpy(X).to(DEVICE)
    length_k = N // K
    scaled_length_k = int(percentage_K * length_k)

    for i in range(K):
        if i == 0:
            idx = np.random.choice(N)
            s1_i = X[idx]
            distance_s1_i = torch.sum((X - s1_i) ** 2, dim = 1)
            distance_sorted_i = torch.argsort(distance_s1_i)
            neigh_i = X[distance_sorted_i[:scaled_length_k]]
            s2_i = get_s2_PCA(neigh_i, s1_i)
            s = torch.cat((s1_i, s2_i), 0)
            s = s.reshape(1, -1)
            #print("shape of s", s.shape)
        else:
            print("shape of s", s.shape)
            dist_to_s = pairwise_distances_segments(X, s)
            dist_min, idx_min = torch.min(dist_to_s, 1)
            prob_X_torch = dist_min / torch.sum(dist_min)
            prob_X = prob_X_torch.cpu().detach().numpy()
            idx_i = np.random.choice(N, p = prob_X)
            s1_i = X[idx_i]
            distance_s1_i = torch.sum((X - s1_i) ** 2, dim = 1)
            distance_sorted_i = torch.argsort(distance_s1_i)
            neigh_i =  X[distance_sorted_i[:scaled_length_k]]
            s2_i = get_s2_PCA(neigh_i, s1_i)
            s12_i = torch.cat((s1_i, s2_i), 0)
            s12_i = s12_i.reshape(1, -1)
            s = torch.cat((s, s12_i), 0)

    return s

# def get_s2_PCA(neigh, s1):
#     pca = PCA(n_components=2)
#     pca.fit(neigh)
#     c_std = 2.0 + np.random.rand()
#     length_s = c_std * np.sqrt(pca.explained_variance_[0])
#     v_dir = pca.components_[0]
#     s2_pos = s1 + length_s * v_dir
#     s2_neg = s1 - length_s * v_dir
#
#     # check which direction is more convenient
#     s_pos = torch.from_numpy(np.hstack((s1, s2_pos)).reshape(1, -1))
#     s_neg = torch.from_numpy(np.hstack((s1, s2_neg)).reshape(1, -1))
#     neigh_torch = torch.from_numpy(neigh)
#     d_pos = pairwise_distances_segments(neigh_torch, s_pos)
#     d_neg = pairwise_distances_segments(neigh_torch, s_neg)
#     if torch.sum(d_pos) < torch.sum (d_neg):
#         s2 = s2_pos
#     else:
#         s2 = s2_neg
#
#     return s2
#
# def PCA_proporcional_dist(X, K, percentage_K):
#
#     N = X.shape[0]
#     X_torch = torch.from_numpy(X)
#     length_k = N // K
#     scaled_length_k = int(percentage_K * length_k)
#
#     for i in range(K):
#         if i == 0:
#             idx = np.random.choice(N)
#             s1_i = X[idx]
#             distance_s1_i = np.sum((X - s1_i) ** 2, axis=1)
#             distance_sorted_i = np.argsort(distance_s1_i)
#             neigh_i = X[distance_sorted_i[:scaled_length_k]]
#             s2_i = get_s2_PCA(neigh_i, s1_i)
#             s = np.hstack((s1_i, s2_i))
#             s = s.reshape(1, -1)
#             #print("shape of s", s.shape)
#         else:
#             s_torch = torch.from_numpy(s)
#             #print("Shape of s_torch", s_torch.shape)
#             dist_to_s = pairwise_distances_segments(X_torch, s_torch)
#             dist_min, idx_min = torch.min(dist_to_s, 1)
#             prob_X_torch = dist_min / torch.sum(dist_min)
#             prob_X = prob_X_torch.cpu().cpu().detach().numpy()
#             idx_i = np.random.choice(N, p = prob_X)
#             s1_i = X[idx_i]
#             distance_s1_i = np.sum((X - s1_i) ** 2, axis=1)
#             distance_sorted_i = np.argsort(distance_s1_i)
#             neigh_i =  X[distance_sorted_i[:scaled_length_k]]
#             s2_i = get_s2_PCA(neigh_i, s1_i)
#             s12_i = np.hstack((s1_i, s2_i))
#             s = np.vstack((s, s12_i))
#
#     return s

def PCA_furthest(X, K, percentage_K):
    N = X.shape[0]
    length_k = N // K
    scaled_length_k = int(percentage_K * length_k)

    for i in range(K):
        if i == 0:
            idx = np.random.choice(N)
            s1_i = X[idx]
            distance_s1_i = np.sum((X - s1_i) ** 2, axis=1)
            distance_sorted_i = np.argsort(distance_s1_i)
            neigh_i = X[distance_sorted_i[:scaled_length_k]]
            s2_i = get_s2_PCA(neigh_i, s1_i)
            s = np.hstack((s1_i, s2_i))
        else:
            s1_range_i = distance_sorted_i[-length_k:]
            idx_i = np.random.choice(length_k)
            s1_i = X[s1_range_i[idx_i]]
            X = X[distance_sorted_i[length_k:]]
            distance_s1_i = np.sum((X - s1_i) ** 2, axis=1)
            distance_sorted_i = np.argsort(distance_s1_i)
            neigh_i =  X[distance_sorted_i[:scaled_length_k]]
            s2_i = get_s2_PCA(neigh_i, s1_i)
            s12_i = np.hstack((s1_i, s2_i))
            s = np.vstack((s, s12_i))

    return s


def PCA_kdivision(X, K, percentage_K):

    N = X.shape[0]
    idx = np.random.choice(N)
    length_k = N // K
    scaled_length_k = int(percentage_K * length_k)
    s1 = X[idx]
    distance_s1 = np.sum((X - s1) ** 2, axis=1)
    distance_sorted = np.argsort(distance_s1)
    neigh = X[distance_sorted[:scaled_length_k]]

    s2 =  get_s2_PCA(neigh, s1)
    s = np.hstack((s1, s2))

    for i in range(1, K):
        s1_range_i = distance_sorted [i * length_k: (i + 1) * length_k]
        idx_i = np.random.choice(length_k)
        s1_i = X[s1_range_i[idx_i]]
        distance_s1_i = np.sum((X - s1_i) ** 2, axis=1)
        distance_sorted_i = np.argsort(distance_s1_i)
        neigh_i = X[distance_sorted_i[:scaled_length_k]]
        s2_i = get_s2_PCA(neigh_i, s1_i)
        s12_i = np.hstack((s1_i, s2_i))
        s = np.vstack((s, s12_i))

    return s
# libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics import mean_squared_error
from helpers import plot_X2D_visualization


def k_means(X, centers_init, n_clusters, verbose = False, plot = False):
    """
    Runs k-means given the data (numpy array) and the number of clusters to consider.
    Arguments:
        X: data (numpy array) to apply k-means.
        n_clusters: number of clusters to consider.
        verbose: boolean variable to use verbose mode.
        plot: boolean variable to plot the clusterized points or not.
    Outputs:
        centers: centroids of the n_clusters clusters.
        labels: labels of the data as a result of k-means.
    """
    if centers_init is None:
        kmeans = KMeans(n_clusters=n_clusters, init='random').fit(X)
    else:
        kmeans = KMeans(n_clusters=n_clusters, init = centers_init).fit(X)
    labels = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    if verbose:
        for i in range(n_clusters):
            print("Cluster: {}, centroid: {}: ".format(i, centers[i]))

    return centers, labels

def visualize_kmeans(writer, X, n_clusters, title_fig, title_plot):
    """

    :param writer:
    :param X:
    :param n_clusters:
    :param title_fig:
    :param title_plot:
    :return:
    """
    centers, labels = k_means(X, n_clusters = n_clusters)
    title = title_fig
    writer.add_figure(title_plot,
                      plot_X2D_visualization(X, labels, title=title, num_classes = n_clusters, cluster_centers=centers))

    return None


def k_medians_manhattan(X, n_clusters, cluster_centers_init=None, max_iter=1000, tol=1e-6, verbose=False):
    """
    Inspired to some degree by the implementation found in: https://gist.github.com/mblondel/1451300.
    Implements the algorithm k-medians by using the manhattan distance.
    Arguments:
        X: data (size: (n, d), n = number of points, d = dimension of the poitns)
            to apply the k-medians algorithm.
        n_clusters: number of (supposed) clusters.
        cluster_centers_init: initial guess of the centers of the clusters if any.
        max_iter: maximum number of iterations of the algorithm.
        tol: tolerance of minimum variation between the previous and current iteration.
        verbose: boolean variable to use verbose mode.

    Outputs:
        cluster_centers: cluster centers found by the algorithm.
        labels: labels assigned according to the minimum manhattan distance
            to the cluster centers.
    """
    if cluster_centers_init is not None:
        cluster_centers = cluster_centers_init
    else:
        idx_init = np.random.permutation(X.shape[0])[:n_clusters]
        cluster_centers = X[idx_init, :]

    for i in range(max_iter):
        dist = manhattan_distances(X, cluster_centers)
        labels = dist.argmin(axis=1)
        cluster_centers_old = cluster_centers.copy()
        if verbose:
            print("iter = {}".format(i))
            print("cluster_centers =\n{}".format(cluster_centers))
            print("labels = {}\n".format(labels));
        for j in range(n_clusters):
            idx = np.where(labels == j)
            cluster_centers[j] = np.median(X[idx], axis=0)

        if mean_squared_error(cluster_centers, cluster_centers_old) < tol:
            break

    return cluster_centers, labels

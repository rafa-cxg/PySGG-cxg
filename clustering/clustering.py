# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle
import time

# import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from functools import reduce
ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
#by cxg
def plot_t_sne(feature,ori_label,cluster_label):
    data_dir = 'clustering/'
    if os.path.isfile(data_dir+'t-sne_result_3.npy')==False:
        tsne = TSNE(n_iter=250, verbose=1, n_components=3)
        tsne_results = tsne.fit_transform(feature)
        print(tsne_results.shape)
        np.save((data_dir+'t-sne_result_3.npy'), tsne_results)
    else:
        tsne_results=np.load(data_dir+'t-sne_result_3.npy')
    # Create the figure
    fig = plt.figure(figsize=(30, 30))
    # ax = fig.add_subplot(1, 1, 1, title='TSNE')
    #
    # # Create the scatter
    # ax.scatter(
    #     x=tsne_results[:, 0],
    #     y=tsne_results[:, 1],
    #     c=cluster_label,
    #     cmap=plt.cm.get_cmap('Paired'),
    #     alpha=0.4,
    #     s=0.5)
    sns.scatterplot(tsne_results[:, 0], tsne_results[:, 1], hue=cluster_label,
                    palette='Set1', s=100, alpha=0.6).set_title('Cluster Vis tSNE Scaled Data', fontsize=15)

    plt.savefig(data_dir+'tsne.png', dpi=300)
    plt.show()

    Scene = dict(xaxis=dict(title='tsne1'), yaxis=dict(title='tsne2'), zaxis=dict(title='tsne3'))
    # labels = labels_tsne_scale
    # trace = go.Scatter3d(x=clusters_tsne_scale.iloc[:, 0], y=clusters_tsne_scale.iloc[:, 1],
    #                      z=clusters_tsne_scale.iloc[:, 2], mode='markers',
    #                      marker=dict(color=labels, colorscale='Viridis', size=10, line=dict(color='yellow', width=5)))
    # layout = go.Layout(margin=dict(l=0, r=0), scene=Scene, height=1000, width=1000)
    # data = [trace]
    # fig = go.Figure(data=data, layout=layout)
    fig.show()
def pil_loader(path):
    """Loads an image.
    Args:
        path (string): path to image file
    Returns:
        Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx][0]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path, pseudolabel))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel = self.imgs[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, 50, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def make_graph(xb, nnn):
    """Builds a graph of nearest neighbors.
    Args:
        xb (np.array): data
        nnn (int): number of nearest neighbors
    Returns:
        list: for each data the list of ids to its nnn nearest neighbors
        list: for each data the list of distances to its nnn NN
    """
    N, dim = xb.shape

    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()

    # L2
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    index.add(xb)
    D, I = index.search(xb, nnn + 1)
    return I, D


def cluster_assign(images_lists, dataset):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    t = transforms.Compose([transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)


def faiss_run_kmeans(x, nmb_clusters, verbose=True):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x[:,:-1].shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    # res = faiss.StandardGpuResources()
    # flat_config = faiss.GpuIndexFlatConfig()
    # flat_config.useFloat16 = False
    # flat_config.device = 0
    index = faiss.IndexFlatL2(d)
    y = x[:, -1]  # instance?????????rel label
    x=x[:,:-1].astype('float32')

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    clustering_label=np.squeeze(I)
    stats = clus.iteration_stats
    losses = np.array([#??????=niter
        stats.at(i).obj for i in range(stats.size())
    ])
    centroids=clus.centroids
    centroids = np.array([
        centroids.at(i) for i in range(centroids.size())#size????????????306
    ])
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))
        print("show t-sne result")
        plot_t_sne(x,y,clustering_label)


    return [int(n[0]) for n in I], losses[-1]


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def faiss_cluster(self, data, verbose=True):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        # xb = preprocess_features(data,data.shape[1]) #???????????????????????????????????????????????????Norm??????
        xb=data
        # cluster the data
        I, loss = faiss_run_kmeans(xb, self.k, verbose)#I: list 0f cluster number. len:40000
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)#list,??????????????????cluster?????????instance??????
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss

    def sklearn_cluster(self, data, verbose=True):
        x=data[:,:-1]
        y=data[:,-1]
        end = time.time()
        # KMeans(algorithm='auto', copy_x=True, init='random', max_iter=300,
        #        n_clusters= self.k, n_init=1000, n_jobs=None, precompute_distances='auto',
        #        random_state=45, tol=0.0001, verbose=0)
        km = KMeans(algorithm='auto', copy_x=True, init='random', max_iter=300,
               n_clusters= self.k, n_jobs=None, precompute_distances='auto',
               random_state=45, tol=0.0001, verbose=0)
        shit=np.any(np.isnan(x))
        km.fit(x)
        print("Adjusted Rand Score: {0:.3f}".format(adjusted_rand_score(y, km.labels_)))

        print("Inertia: {0:.1f}".format(km.inertia_))
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        matchs = reduce(np.union1d, y)
        evaluater={}
        for match in matchs:
            mask=np.nonzero(y==match)
            total=np.count_nonzero(y==match)
            for i in range(0,self.k):
                evaluater[i]=np.count_nonzero(km.labels_[mask]==i)/total*100
            array = np.array(list(evaluater.items()))
            idx=np.argmax(array[:,-1],-1)
            print('relation {} belong to {} cluster for {}%, in {} samples'.format(match,idx, evaluater[idx],total))
        if verbose:

            print("show t-sne result")
            plot_t_sne(x, y, km.labels_)

def make_adjacencyW(I, D, sigma):
    """Create adjacency matrix with a Gaussian kernel.
    Args:
        I (numpy array): for each vertex the ids to its nnn linked vertices
                  + first column of identity.
        D (numpy array): for each data the l2 distances to its nnn linked vertices
                  + first column of zeros.
        sigma (float): Bandwidth of the Gaussian kernel.

    Returns:
        csr_matrix: affinity matrix of the graph.
    """
    V, k = I.shape
    k = k - 1
    indices = np.reshape(np.delete(I, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(d):
        return np.exp(-d / sigma**2)

    exp_ker = np.vectorize(exp_ker)
    res_D = exp_ker(D)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def run_pic(I, D, sigma, alpha):
    """Run PIC algorithm"""
    a = make_adjacencyW(I, D, sigma)
    graph = a + a.transpose()
    cgraph = graph
    nim = graph.shape[0]

    W = graph
    t0 = time.time()

    v0 = np.ones(nim) / nim

    # power iterations
    v = v0.astype('float32')

    t0 = time.time()
    dt = 0
    for i in range(200):
        vnext = np.zeros(nim, dtype='float32')

        vnext = vnext + W.transpose().dot(v)

        vnext = alpha * vnext + (1 - alpha) / nim
        # L1 normalize
        vnext /= vnext.sum()
        v = vnext

        if i == 200 - 1:
            clust = find_maxima_cluster(W, v)

    return [int(i) for i in clust]


def find_maxima_cluster(W, v):
    n, m = W.shape
    assert (n == m)
    assign = np.zeros(n)
    # for each node
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for l in range(l0, l1):
            j = W.indices[l]
            vi = W.data[l] * (v[j] - v[i])
            if vi > best_vi:
                best_vi = vi
                pointers[i] = j
    n_clus = 0
    cluster_ids = -1 * np.ones(n)
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus = n_clus + 1
    for i in range(n):
        # go from pointers to pointers starting from i until reached a local optim
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]

        assign[i] = cluster_ids[current_node]
        assert (assign[i] >= 0)
    return assign


class PIC(object):
    """Class to perform Power Iteration Clustering on a graph of nearest neighbors.
        Args:
            args: for consistency with k-means init
            sigma (float): bandwidth of the Gaussian kernel (default 0.2)
            nnn (int): number of nearest neighbors (default 5)
            alpha (float): parameter in PIC (default 0.001)
            distribute_singletons (bool): If True, reassign each singleton to
                                      the cluster of its closest non
                                      singleton nearest neighbors (up to nnn
                                      nearest neighbors).
        Attributes:
            images_lists (list of list): for each cluster, the list of image indexes
                                         belonging to this cluster
    """

    def __init__(self, args=None, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True):
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons

    def cluster(self, data, verbose=False):
        end = time.time()

        # preprocess the data
        xb = preprocess_features(data)

        # construct nnn graph
        I, D = make_graph(xb, self.nnn)

        # run PIC
        clust = run_pic(I, D, self.sigma, self.alpha)
        images_lists = {}
        for h in set(clust):
            images_lists[h] = []
        for data, c in enumerate(clust):
            images_lists[c].append(data)

        # allocate singletons to clusters of their closest NN not singleton
        if self.distribute_singletons:
            clust_NN = {}
            for i in images_lists:
                # if singleton
                if len(images_lists[i]) == 1:
                    s = images_lists[i][0]
                    # for NN
                    for n in I[s, 1:]:
                        # if NN is not a singleton
                        if not len(images_lists[clust[n]]) == 1:
                            clust_NN[s] = n
                            break
            for s in clust_NN:
                del images_lists[clust[s]]
                clust[s] = clust[clust_NN[s]]
                images_lists[clust[s]].append(s)

        self.images_lists = []
        for c in images_lists:
            self.images_lists.append(images_lists[c])

        if verbose:
            print('pic time: {0:.0f} s'.format(time.time() - end))
        return 0

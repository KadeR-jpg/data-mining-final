import queue
import numpy as np

# -> Inputs
# Data Matrix: array
# Number of clusters: kClust
# Covergence Param: eps
# Initial Means: iMeans
# <- Returns
# Representative means, clusters found
# Notes: If the distance between a point and more than
# one representative(mean), then assign the point to the
# mean corresponding to the cluster w/lowest index.

# %%


class kMeans():
    def __init__(self, data, kClust, iters, eps, centroids):
        self.data = data
        self.kClust = kClust
        self.iters = iters
        self.eps = eps
        self.centroids = centroids
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(kClust)]
        # Mean ft vector for each cluster

    def predict(self):
        self.nSamp, self.nFeat = self.data.shape
        self.clusters = self.get_clusters(self.centroids)

        # make them better

        for _ in range(self.iters):
            # update clusters
            self.clusters = self.get_clusters(self.centroids)

            # update centr koids
            oldCent = self.centroids
            self.centroids = self.get_centroids(self.clusters)
            if self.converge(oldCent, self.centroids):
                break

        return self.get_cluster_label(self.clusters)

    def get_cluster_label(self, cluster):
        labels = np.empty(self.nSamp)
        for cluster_index, cluster in enumerate(cluster):
            for point_index in cluster:
                labels[point_index] = cluster_index
        return labels

        # check if means match(converge)

    def get_clusters(self, centroids):
        cluster = [[] for _ in range(self.kClust)]
        for i, point in enumerate(self.data):
            cent_index = self.closest_centroid(point, centroids)
            cluster[cent_index].append(i)
        return cluster

    def closest_centroid(self, sample, centroids):
        dist = [np.linalg.norm(sample - point) for point in centroids]
        closest = np.argmin(dist)
        return closest

    def get_centroids(self, clusters):
        cents = np.zeros((self.kClust, self.nFeat))
        for cluster_index, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.data[cluster], axis=0)
            cents[cluster_index] = cluster_mean
        return cents

    def converge(self, oldC, newC):
        dist = [np.linalg.norm(oldC[i] - newC[i]) for i in range(self.kClust)]
        return sum(dist) == self.eps

    def get_mean(self):
        return np.round(self.centroids, decimals=5)


# -> Inputs
# Data Matrix: array
# Minimum num of points: minpts
# Epsilon: eps
# <- Returns
# Clusters found where each point is labeled as
# either a noise point, border point or core point
# def dbScan(array, minpts, eps):

# %%


class DBSCAN():
    def __init__(self, data, minpts, eps):
        self.data = data
        self.minpts = minpts
        self.eps = eps
        self.core = -1
        self.border = -2
        self.labels = [0 for _ in range(len(self.data))]
        self.core_index = []
        self.border_index = []
        self.noise = []
        self.clusters = 0

    def predict(self):
        current = []
        core_pt = []
        border_pt = []
        for point in range(len(self.data)):
            current.append(self.get_neighbors(point))

        for point in range(len(current)):
            if(len(current[point])) >= self.minpts:
                self.labels[point] = self.core
                core_pt.append(point)
            else:
                border_pt.append(point)
        for point in border_pt:
            for x in current[point]:
                if x in core_pt:
                    self.labels[point] = self.border
                    break

        cluster = 0

        for i in range(len(self.labels)):
            search_q = queue.Queue()
            if(self.labels[i] == self.core):
                self.labels[i] = cluster
                for x in current[i]:
                    if(self.labels[x] == self.core):
                        search_q.put(x)
                        self.labels[x] = cluster
                    elif(self.labels[x] == self.border):
                        self.labels[x] = cluster
                while not search_q.empty():
                    neighbors = current[search_q.get()]
                    for this in neighbors:
                        if(self.labels[this] == self.core):
                            self.labels[this] = cluster
                            search_q.put(this)
                        if(self.labels[this] == self.border):
                            self.labels[this] = cluster
                cluster += 1

        self.core_index.append(core_pt)
        self.border_index.append(border_pt)
        self.clusters = cluster
        self.get_noise()
        # self.get_border()
        return self.labels, cluster

    def get_neighbors(self, point_index):
        neighbors = []
        for i in range(len(self.data)):
            if np.linalg.norm(self.data[i] - self.data[point_index]) <= self.eps:
                neighbors.append(i)
        return neighbors

    # def get_border(self):
    #     cluster = 0
    #     noise = []
    #     for i in range(len(self.labels)):
    #         if(self.labels[i] == cluster and i not in self.core_index[0]):
    #             noise.append(i)
    #     cluster += 1
    #     self.noise.append(noise)
    def get_noise(self):
        for i in range(len(self.labels)):
            if(self.labels[i] == 0 and i not in self.core_index[0]):
                self.noise.append(i)

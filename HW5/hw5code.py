# %%
from matplotlib.colors import Normalize
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.core.arrayprint import printoptions
from scipy.sparse.linalg import eigen
import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans, DBSCAN

# Question 1 code from here to line 15
mu = np.array([0, 0])
Sigma = np.array([[1, 0], [0, 1]])
X1, X2 = np.random.multivariate_normal(mu, Sigma, 1000).T
D = np.array([X1, X2]).T
col1 = D[:, 0]
col2 = D[:, 1]
row1 = D[0, :]
row2 = D[1, :]
# origScatter = plt.scatter(col1, col2, alpha=.5, c='red')
# ---- End of question 1 code
# Start of question 2 code
X = D - np.mean(D, 0)
theta = math.pi/4

R = np.array([
    [math.cos(theta), -math.sin(theta)],
    [math.sin(theta), math.cos(theta)]
])
S = np.array([
    [5, 0],
    [0, 2]
])

tMatrix = S.dot(R)
RSD = X.dot(tMatrix)
rsdCol1 = RSD[:, 0]
rsdCol2 = RSD[:, 1]
# rsdScatter = plt.scatter(rsdCol1, rsdCol2, c='grey', marker='o', alpha=.4)
# ---- End of question 2
df = pd.DataFrame(RSD)
dfCov = df.cov()
# Start of question 3
x = StandardScaler().fit_transform(RSD)
pca = PCA(n_components=2)
rsdProjected = pca.fit_transform(x)


x1 = rsdProjected[:, 0]
x2 = rsdProjected[:, 1]

# ---PCA by handish...
df = pd.DataFrame(rsdProjected)
pcaCov = df.cov()
# eigenVal, eigenVect = np.linalg.eig(dfCov)
# eigenSort = np.argsort(eigenVal)
# featVect = eigenVect[:, eigenSort[1:]]
# rowVect = featVect.T
# rowData = dfMean.T
# dfReduced = np.matmul(rowVect, rowData)
# fit1 = dfReduced.T[0]
# fit2 = dfReduced.T[1]
# pcaPlot = plt.scatter(x1, x2, c='purple', alpha=.5, marker='o')
# plt.show()
# plt.legend((origScatter, rsdScatter, pcaPlot),
#            ('Original Data', 'Linear Transformed Data', 'PCA Data'),
#            loc='upper right')
# %%
boston = load_boston()
D = boston['data']
Y = StandardScaler().fit_transform(D)
bostonPCA = PCA()
bostonPro = bostonPCA.fit_transform(Y)
bostonCov = bostonPCA.get_covariance()
b_X = bostonPro[:, 0]
b_Y = bostonPro[:, 1]
# pcaSum = np.cumsum(bostonPCA.explained_variance_ratio_)
# plt.plot(pcaSum, scalex=1)
# plt.axhline(y=.9, c='m')
# plt.axvline(x=6.5, c='m')
# plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# kmeans = KMeans(n_clusters=2, n_init=10, max_iter=300,
#                 random_state=0, init='random')
# fit_KM = kmeans.fit_predict(bostonPro)
# plt.scatter(
#     bostonPro[fit_KM == 0, 0], bostonPro[fit_KM == 0, 1],
#     marker='o', edgecolors='black', alpha=.5, c='purple',
#     label='cluster 1'
# )
# plt.scatter(
#     bostonPro[fit_KM == 1, 0], bostonPro[fit_KM == 1, 1],
#     edgecolors='black', alpha=.5, c='teal',
#     marker='o',
#     label='cluster 2'
# )
# plt.scatter(
#     kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#     s=200, marker='*', c='white', edgecolors='black',
#     label='centroids'
# )
# plt.legend(scatterpoints=1)
# plt.show()
db = DBSCAN(eps=2.65, min_samples=65).fit(bostonPro)
labels = db.labels_
print(labels)
color = {}
color[0] = 'r'
color[1] = 'g'
color[2] = 'y'
color[-1] = 'w'
cvec = [color[label] for label in labels]
r = plt.scatter(bostonPro[:, 0], bostonPro[:, 1], c='r', alpha=.2)
g = plt.scatter(bostonPro[:, 0], bostonPro[:, 1], c='g', alpha=.2)
k = plt.scatter(bostonPro[:, 0], bostonPro[:, 1], c='w')
plt.scatter(bostonPro[:, 0], bostonPro[:, 1], c = cvec)
plt.legend((r, g, k), ('Cluster 1', 'Cluster 2', 'Noise'))
plt.show()
# %%

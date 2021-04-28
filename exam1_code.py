# %%
import numpy as np
import math
from numpy.core.numeric import NaN
import pandas as pd
from sklearn import preprocessing
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import DBSCAN
import networkx as nx


mat = np.random.rand(3, 2)
df = pd.DataFrame(mat)
min_max = preprocessing.MinMaxScaler()
scaled = min_max.fit_transform(df)
df_norm = pd.DataFrame(scaled)
# print(df_norm.mean(axis=1))
# Question 4
X_1 = np.array([1, 3.4, 3, -5, 11, 3])
X_2 = np.array([-1, -3.5, 2, 2, 0, -2])
# print(np.cov(X_1, X_2))
# End of Question 4
# Question 5
D = np.array([
    [0.3, 21, 5.6],
    [0.4, 1, 5.2],
    [1.8, 4, 5.2],
    [7, 45, 5.1],
    [-0.5, 34, 5.8],
    [0.4, 13, 5.4],
    [1.1, 11, 5.5],
])
# print(D.var())
# End of Question 5

# Question 6
y_1 = np.array([1, -1, 1, 3])
y_2 = np.array([-1, 1, 0, 3])
# print(np.linalg.norm(y_1 - y_2))
# print(np.linalg.norm(y_1 - y_2, ord=1))
# End of question 6

# Question 7
# H = np.array([
#     [3, 'A', 3, 0],
#     [-1, 'B', 2, 1],
#     [2, 'C', 1, 0],
#     [1, 'C', 1, -1],
# ])
# row2 = np.array([-1, 0, 1, 0, 2, 1])
# row3 = np.array([2, 0, 0, 1, 1, 0])
# print(np.linalg.norm(row2 - row3))
# print(fitH)
# End of question 7
# Question 8
D = np.array([
    [3, 4, 0],
    [NaN, NaN, 1],
    [NaN, 0, 2.2],
    [3.4, 1, NaN],
    [NaN, 0, 2],
])
df = pd.DataFrame(D)
# print(df.ffill(axis=0))
# End of question 8
# Question 9
B = np.array([
    [1, -1, 2],
    [4, 2, 1],
    [-5, -3, 7],
    [-3, 2, -1],
    [7, -1, 0],
])
# print(np.round(np.cov(B), decimals=2))
# End of question 9
# Question 10
C = np.array([
    [0, 1, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 1, 1, 0, 0],
])
c_G = nx.from_numpy_array(C)
# nx.draw(c_G, with_labels=True)
# print(nx.clustering(c_G))
# End of question 10
# Question 11
nodes = [i for i in range(5)]
edgeList = ([
    (0, 2),
    (0, 3),
    (1, 0),
    (2, 1),
    (2, 4),
    (3, 2),
    (4, 3),
    (4, 3),
])
G = nx.DiGraph()
for node in nodes:
    G.add_node(node)
G.add_edges_from(edgeList)
# nx.draw_spectral(G, with_labels=True)
start = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
}
centrality = nx.eigenvector_centrality(G, nstart=start)
# End of question 11
# Question 15
theta = math.pi/3
S = np.array([
    [3, 0],
    [0, 2]
])
R = np.array([
    [math.cos(theta), -math.sin(theta)],
    [math.sin(theta), math.cos(theta)]
])
x = np.array([[1], [1]])
# print(x)
# End of question 15
# Question 18
E = np.array([
    [4, 1],
    [4.9, 5.1],
    [-2, 2],
    [-3, 1],
    [4.5, 4],
    [4, 4.5],
    [-1.1, 1.8],
    [-1, 6.7],
    [3, 4.2],
    [-2, 0.9],
    [5.7, 3.8],
])
cluster1 = []
cluster2 = []
means = [[2, -1], [1, 5]]
for node in E:
    dist1 = np.linalg.norm(node - means[0])
    dist2 = np.linalg.norm(node - means[1])
    if dist1 > dist2:
        cluster2.append(node)
    else:
        cluster1.append(node)
# End of question 18
# Question 19
cluster = DBSCAN(eps=3, min_samples=5).fit(E)
# End of question 19
# Random Scratch work
norm = preprocessing.normalize(E)
print(norm)

# %%

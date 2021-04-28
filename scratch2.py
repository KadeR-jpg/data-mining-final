# %%
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.matrixlib.defmatrix import matrix


def avg(arr):
    avg = 0
    size = len(arr)
    for i in range(size):
        avg += arr[i]
    return avg/size


def mCentered(matrix):
    matMean = np.array([0.0])
    matMean.resize(len(matrix), len(matrix[0]))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matMean[i][j] = matrix[i][j] - avg(matrix[:, j])
    return matMean


def covar(mat1, mat2):
    avgONE = avg(mat1)
    avgTWO = avg(mat2)

    covar = 0
    for i in range(len(mat1)):
        covar += (mat1[i] - avgONE) * (mat2[i] - avgTWO)
    return covar/(len(mat1) - 1)


def covarMat(mat):
    covarM = np.array([0.0])
    covarM.resize(len(mat[0]), len(mat[0]))
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            covarM[i][j] = covar(mat[:, i], mat[:, j])
    return covarM


def rangeNorm(mat):
    rMat = np.array([0.0])
    rMat.resize(len(mat), len(mat[0]))
    for i in range(len(mat[0])):
        for j in range(len(mat)):
            rMat[i][j] = (mat[j][i] - np.min(mat[:, i])) / \
                (np.max(mat[:, i]) - np.min(mat[:, i]))
    return rMat


A = np.array([
    [math.sqrt(3)/2, -0.5],
    [0.5, math.sqrt(3)/2]
])
D = np.array([
    [1, 1],
    [1, 2],
    [3, 4],
    [-1, -1],
    [-1, 1],
    [1, -2],
    [2, 2],
    [2, 3]
])

# plt.show(plt.scatter(D[:,0], D[:, 1]))
# t = A.dot(D.transpose()).transpose()
# plt.scatter(D[:,0], D[:, 1], label = 'Original Data')
# plt.scatter(t[:,0], t[:, 1], label = 'Transformed Data')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.legend(loc = 'upper left')
# plt.title('Orig vs Transformed Data')
# plt.show()

# cent = mCentered(D)

# plt.scatter(cent[:, 0], cent[:, 1] , label = 'Mean centered')
# plt.scatter(D[:, 0], D[:, 1], label = 'Origianl Data')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.legend(loc = 'upper left')
# plt.title("Mean Centered vs Orig Data")
# plt.show()

# %%

# %%
from contextlib import suppress
from math import sqrt
import numpy as np
from numpy import ma
import pandas as pd
import csv
import matplotlib.pyplot as plt

def multiVarMean(twoDarr):
    avgVector = []
    avg = 0
    for x in range(len(twoDarr[0])):
        for y in range(len(twoDarr)):
            avg += twoDarr[y][x]
        avgVector += [avg / len(twoDarr)]
    return avgVector


def mean_2d(array):
    size = np.size(array)
    total = 0
    for element in np.nditer(array):
        total += element

    return total/size


def samp_covar(arr1, arr2):
    size = np.size(arr1)
    meanX = mean_2d(arr1)
    meanY = mean_2d(arr2)
    covar = 0
    for x in range(size):
        covar += ((arr1[x] - meanX) * (arr2[x] - meanY))

    return(covar/(size-1))


def correlation(arr1, arr2):
    meanX = mean_2d(arr1)
    meanY = mean_2d(arr2)
    bottom = 0
    for x in range(np.size(arr1)):
        bottom = (sqrt(pow((arr1[x] - meanX), 2))
                  * sqrt(pow((arr2[x] - meanY), 2)))

    correl = samp_covar(arr1, arr2) / bottom
    return correl


def normalize2d_range(inMat):
    normRange = np.array([0.0])
    normRange.resize(len((inMat)), len(inMat[0]))
    elements = len(inMat[0])
    row = len(inMat)
    for x in range(elements):
        for y in range(row):
            normRange[y][x] = (inMat[y][x] - np.min(inMat[:, x])) / \
                (np.max(inMat[:, x]) - np.min(inMat[:, x]))
    return normRange


def normalize2d_std(inMat):
    normStd = np.array([0.0])
    normStd.resize(len(inMat[0]), len(inMat[0]))
    for x in range(len(inMat[0])):
        for y in range(len(inMat)):
            normStd[y][x] = (inMat[y][x] - mean_2d(inMat[:, x])) / \
                np.sqrt(samp_covar(inMat[:, x], inMat[:, x]))
    return normStd


def covar_matrix(inMat):
    covarianceMatrix = np.array([0.0])
    covarianceMatrix.resize(len(inMat[0]), len(inMat[0]))
    for x in range(len(inMat[0])):
        for y in range(len(inMat[0])):
            covarianceMatrix[x][y] = correlation(
                inMat[:, x], inMat[:, y])
    return covarianceMatrix

# Getting a column that contains the categorical attributes and
# label encodes them


def lbl_ncd(inMat):
    strs = []
    # I found this lambda expression online
    # all its doing is retrieving the strings and adding them to
    # the array "strs" if its not already added into it
    [strs.append(str(i)) for i in inMat if str(i) not in strs]
    ofStrs = np.array(range(0, len(strs)))
    # Zipping our two arrays together to make a matrix
    strDict = dict((zip(strs, ofStrs)))
    inMat[:] = [float(strDict.get(e, '')) for e in inMat]
    return inMat


with open('forestfires.csv', 'r') as csvfile:

    npData = list(csv.reader(csvfile))
    npData = np.delete(npData, (0), axis=0)
    # Label encoding the months
    lbl_ncd(npData[:, 2])
    # Label encoding the days
    lbl_ncd(npData[:, 3])
    # Using asfarray to simply convert the ints to floats and then storing it as an array.
    npData = np.asfarray(npData)
    npData = normalize2d_range(npData)
    multiVMean = multiVarMean(npData)
    covarMatrix = covar_matrix(npData)
    x = npData[:, 0]
    y = npData[:, 1]
    month = npData[:, 2]
    FFMC = npData[:, 4]
    dmc = npData[:, 5]
    dc = npData[:, 6]
    ISI = npData[:, 7]
    temp = npData[:, 8]
    RH = npData[:, 9]
    wind = npData[:, 10]
    rain = npData[:, 11]
    area = npData[:, 12]



    plt.scatter(x, ISI, marker='.')
    plt.xlabel('X')
    plt.ylabel('Initial Spread Index')
    plt.title('X Coord vs Initial Spread Index')
    # plt.scatter(FFMC, ISI)
    # plt.xlabel('Fine Fuel Moisture Code')
    # plt.ylabel('Initial Spread Index')

    # with np.printoptions(precision=3, suppress=True):
        # print(multiVMean)
        # print(np.asfarray(covarMatrix))

# Using pandas because it is easier to get entire columns
# with open('forestfires.csv', 'r') as csvfile:
#     data = csv.reader(csvfile)
#     df = pd.DataFrame(data)
#     # Getting rid of the index row and shifting the named row up by 1
#     newHeader = df.iloc[1]
#     df = df[1:]
#     df.columns = newHeader
#     print(df)

    # df['month'] = pd.to_datetime(df['month'], format='%b').dt.month
    # tempnRain = np.array([df['temp'].astype(float),
    #                       df['wind'].astype(float)])

    # month = np.array(df['month'].astype(float))
    # wind = np.array(df['wind'].astype(float))

    # meanRet = multiVarMean(tempnRain)
    # covarRet = samp_covar(month, wind)
    # print('The mean of the 2d array composed of the ' +
    #       f'Rain and Wind data is {np.round(meanRet, decimals=3)}')
    # print('---------------------------')
    # print(f'The covariance is {covarRet.round(3)}')
    # print(f'The correlation is {correlation(month, wind).round(3)}')
    # print(f'The range normalized matrix of temp and rain is {normalize2d_range(tempnRain).round(3)}')
    # print(f'The standard normilized matrix is {normalize2d_std(tempnRain).round(3)}')


# %%

# %%

# %%

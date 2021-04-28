import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import load_iris
import proj3_main as testOn
##########################
# Kmeans Test OBJ 1
##########################
t1_data = np.array([[-1, 2], [-2, -2], [3, 2], [5, 4.3], [-3, 3],
                    [-3, -1], [5, -3], [3, 4], [2.3, 6.5]])
t1_in_means = [[3.90304975, 2.51839422],
               [-1.21226502, 3.6765421]]
t1_obj = testOn.kMeans(t1_data, 2, 20, 0.000001, t1_in_means)
t1_fit = t1_obj.predict()
t1_exp_fit = [1, 1, 0, 0, 1, 1, 0, 0, 0]
t1_mean = t1_obj.get_mean()
t1_exp_mean = [[3.66,  2.76],
               [-2.25,  0.5]]
##########################
# Kmeans Test OBJ 2
##########################
t2_data = np.array([[-1, 2], [-2, -2], [3, 2], [5, 4.3],
                    [-3, 3], [-3, -1], [5, -3], [3, 4],
                    [2.3, 6.5], [4, 2], [4, 4], [-2.3, 1.5]])
t2_in_means = [[3.67371566, 3.05039513],
               [-2.27144128, 0.58447117],
               [4.50290941, 4.78503096]]
t2_obj = testOn.kMeans(t2_data, 3, 20, 0.000001, t2_in_means)
t2_fit = t2_obj.predict()
t2_exp_fit = [1., 1., 0., 2., 1., 1., 0., 2., 2., 0., 2., 1.]
t2_mean = t2_obj.get_mean()
t2_exp_mean = np.round(
    [[4.,  0.33333333],
     [-2.26,  0.7],
     [3.575,  4.7]], decimals=5)
##########################
# Kmeans Test OBJ 3
##########################
X, y = make_blobs(n_samples=300, centers=3, random_state=35)
t3_means = [[3.75224869, -2.9680029],
            [-1.45630451, -1.65071389],
            [-3.16968936, -1.47368135]]
t3_obj = testOn.kMeans(X, 3, 20, 0.00000001, t3_means)
t3_fit = t3_obj.predict()
t3_mean = t3_obj.get_mean()
t3_exp_mean = np.round([[6.20253055, -7.98535796],
                        [-0.69349908, -3.83286075],
                        [-5.36633913, -4.62487197]], decimals=5)
t3_exp_fit = [0., 2., 0., 1., 0., 0., 2., 0., 1., 1., 2., 2., 0., 0., 2., 2., 1.,
              1., 1., 0., 1., 1., 1., 2., 0., 0., 0., 2., 1., 1., 0., 1., 0., 2.,
              2., 0., 1., 1., 2., 2., 2., 1., 0., 2., 2., 1., 2., 0., 2., 1., 0.,
              0., 2., 2., 0., 1., 2., 1., 2., 2., 1., 1., 2., 1., 2., 1., 0., 1.,
              1., 2., 2., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 2., 1., 1.,
              0., 2., 2., 2., 0., 0., 1., 2., 0., 1., 0., 0., 2., 2., 2., 2., 0.,
              2., 0., 2., 2., 0., 0., 1., 0., 2., 1., 0., 0., 0., 1., 0., 2., 2.,
              2., 2., 1., 1., 2., 1., 2., 1., 0., 0., 2., 1., 2., 1., 2., 1., 0.,
              0., 2., 1., 1., 0., 0., 1., 1., 0., 1., 1., 2., 1., 1., 0., 0., 1.,
              0., 0., 2., 0., 2., 2., 1., 1., 1., 0., 2., 0., 0., 2., 1., 2., 2.,
              2., 0., 2., 1., 2., 2., 0., 2., 1., 0., 0., 2., 0., 0., 1., 2., 0.,
              0., 1., 0., 2., 0., 0., 0., 0., 0., 1., 1., 2., 0., 1., 0., 2., 1.,
              2., 2., 0., 2., 1., 2., 0., 0., 0., 0., 0., 2., 1., 2., 2., 1., 2.,
              2., 1., 2., 0., 0., 1., 0., 0., 2., 2., 0., 2., 0., 1., 1., 2., 0.,
              0., 1., 0., 1., 2., 0., 2., 2., 1., 0., 1., 2., 1., 2., 0., 2., 2.,
              1., 2., 0., 2., 1., 0., 0., 0., 2., 0., 0., 1., 2., 2., 1., 1., 2.,
              2., 1., 1., 1., 1., 1., 0., 1., 0., 1., 0., 0., 2., 0., 2., 0., 0.,
              1., 1., 2., 2., 1., 2., 0., 1., 1., 1., 1.]
##########################
# Kmeans Test OBJ 4
##########################
moons, y = make_moons(n_samples=200, noise=.06, random_state=4)
t4_in_means = [[1.52642034, -0.4699234],
               [0.42680618, 0.51168439]]
t4_obj = testOn.kMeans(moons, 2, 20, 0.00001, t4_in_means)

t4_fit = t4_obj.predict()
t4_exp_fit = [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1,
              0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0,
              0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
              0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0,
              1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1,
              0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0,
              0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
              0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1,
              0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
              0, 1]
t4_mean = t4_obj.get_mean()
t4_exp_mean = np.round([[1.2412556, -0.11496583],
                        [-0.15971746,  0.56669345]], decimals=5)

##########################
# Kmeans Test OBJ 5
##########################
X_iris = load_iris()['data']
t5_in_mean = [[5.24513354, 3.4487186, 6.59654982, 0.36294203],
              [7.08970814, 4.3925459,  6.67452647, 0.13835156],
              [7.08253763, 2.14773688, 2.50271098, 0.91692539]]
t5_obj = testOn.kMeans(X_iris, 3, 20, 0.00001, t5_in_mean)
t5_fit = t5_obj.predict()
t5_exp_fit = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
              2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
              2, 2, 2, 2, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
              1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1,
              1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
t5_mean = t5_obj.get_mean()
t5_exp_mean = np.round([[5.9016129, 2.7483871, 4.39354839, 1.43387097],
                        [6.85, 3.07368421, 5.74210526, 2.07105263],
                        [5.006, 3.428, 1.462, 0.246]], decimals=5)

##########################
# DBSCAN Test OBJ 1
# -> (Data, minpts, eps)
##########################
X, y = make_blobs(n_samples=300, centers=3, random_state=35)
d1_obj = testOn.DBSCAN(X, 10, 0.5)
# %%
d1_fit = d1_obj.predict()
d1_core = d1_obj.core_index[0]
d1_border = d1_obj.border_index[0]
d1_noise = d1_obj.noise
# print(d1_border)


d1_exp_core = [1,
               6,
               9,
               10,
               11,
               12,
               15,
               17,
               20,
               21,
               30,
               35,
               36,
               38,
               39,
               51,
               61,
               69,
               74,
               75,
               77,
               79,
               87,
               88,
               94,
               95,
               98,
               103,
               104,
               112,
               115,
               116,
               117,
               119,
               126,
               141,
               143,
               147,
               151,
               155,
               158,
               161,
               169,
               170,
               174,
               175,
               177,
               180,
               181,
               184,
               188,
               191,
               201,
               205,
               208,
               211,
               212,
               214,
               223,
               224,
               236,
               237,
               239,
               242,
               243,
               245,
               247,
               248,
               257,
               258,
               260,
               262,
               263,
               270,
               279,
               287,
               291,
               296,
               299]
d1_exp_noise = [0,
                2,
                3,
                4,
                7,
                8,
                13,
                16,
                19,
                25,
                27,
                28,
                29,
                31,
                32,
                37,
                40,
                41,
                42,
                43,
                44,
                45,
                47,
                52,
                53,
                54,
                58,
                59,
                60,
                65,
                67,
                68,
                71,
                72,
                73,
                78,
                80,
                82,
                83,
                84,
                85,
                89,
                91,
                93,
                100,
                101,
                102,
                106,
                107,
                108,
                109,
                111,
                113,
                118,
                120,
                121,
                124,
                127,
                128,
                130,
                131,
                136,
                139,
                140,
                142,
                144,
                148,
                149,
                150,
                153,
                156,
                157,
                159,
                163,
                165,
                166,
                167,
                168,
                171,
                172,
                173,
                176,
                178,
                179,
                182,
                183,
                185,
                189,
                192,
                193,
                195,
                196,
                197,
                198,
                199,
                200,
                206,
                207,
                209,
                210,
                216,
                218,
                219,
                220,
                221,
                222,
                225,
                227,
                228,
                229,
                230,
                231,
                232,
                233,
                235,
                238,
                241,
                244,
                246,
                249,
                251,
                252,
                254,
                255,
                256,
                261,
                266,
                268,
                269,
                271,
                272,
                273,
                274,
                276,
                278,
                281,
                283,
                284,
                285,
                286,
                288,
                289,
                290,
                293,
                295,
                297,
                298]
d1_exp_border = [5,
 14,
 18,
 22,
 23,
 24,
 26,
 33,
 34,
 46,
 48,
 49,
 50,
 55,
 56,
 57,
 62,
 63,
 64,
 66,
 70,
 76,
 81,
 86,
 90,
 92,
 96,
 97,
 99,
 105,
 110,
 114,
 122,
 123,
 125,
 129,
 132,
 133,
 134,
 135,
 137,
 138,
 145,
 146,
 152,
 154,
 160,
 162,
 164,
 186,
 187,
 190,
 194,
 202,
 203,
 204,
 213,
 215,
 217,
 226,
 234,
 240,
 250,
 253,
 259,
 264,
 265,
 267,
 275,
 277,
 280,
 282,
 292,
 294]
print(d1_noise)
# print(d1_border)
# print(d1_exp_noise)

def test_kMeans():
    assert np.all(t1_fit == t1_exp_fit)
    assert np.all(t1_mean == t1_exp_mean)
    assert np.all(t2_fit == t2_exp_fit)
    assert np.all(t2_mean == t2_exp_mean)
    assert np.all(t3_fit == t3_exp_fit)
    assert np.all(t3_mean == t3_exp_mean)
    assert np.all(t4_fit == t4_exp_fit)
    assert np.all(t4_mean == t4_exp_mean)
    assert np.all(t5_fit == t5_exp_fit)
    assert np.all(t5_mean == t5_exp_mean)


def test_DBSCAN():
    # Test 1
    assert d1_obj.clusters == 3
    assert np.all(d1_core == d1_exp_core)
    # assert np.all(d1_border == d1_exp_border)
    assert np.all(d1_noise == d1_exp_noise)




# test_kMeans()
test_DBSCAN()


# %%

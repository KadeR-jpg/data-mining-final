# %%
import numpy as np
from numpy.lib.function_base import cov
import pandas as pd
x = np.array([[.3, 23, 5.6],
              [.4, 1, 5.2],
              [1.8, 4, 5.2],
              [6, 50., 5.1],
              [-.5, 34, 5.7],
              [.4, 19, 5.4],
              [1.1, 11, 5.5]])

x_p = pd.DataFrame([(.3, 23, 5.6),
                    (.4, 1, 5.2),
                    (1.8, 4, 5.2),
                    (6, 50., 5.1),
                    (-.5, 34, 5.7),
                    (.4, 19, 5.4),
                    (1.1, 11, 5.5)])

# col_3 = x[:, [2]]
# col_2 = x[:, [1]]
# col_1 = x[:, [0]]

# col_1 = x_p[0]
# col_3 = x_p[2]
# correlation  = col_1.corr(col_3)
# covar = x_p.cov()
# cov = np.cov(x, bias=False)
# print(np.round(cov, decimals=3))
## Question 3; four dimensional vector
a = np.array([1, -1, -2, 4])
b = np.array([2, -1, -1, 3])
# print(np.dot(a, b))
## A
A = np.linalg.norm((a-b), ord=2)
## A returns 1.7320508
B = np.linalg.norm((a-b), ord=1)
## B returns 3.0
unitVec1 = a/np.linalg.norm(a)
unitVec2 = b/np.linalg.norm(b)
dotProduct = np.dot(unitVec1, unitVec2)
C = np.rad2deg(np.arccos(dotProduct))
## 20.64 degrees between the vectors


# %%

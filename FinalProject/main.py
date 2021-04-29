#%%
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import euclidean, minkowski
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class KNN:

    def mink_dist(self, x, y):
        dist = 0
        for dimension in range(x):
            dist += np.abs(x[dimension] - y[dimension])**1
        return dist

    def predict(self, X_train, X_test, y, a):
        labels = []
        for row in X_test:
            distance = np.array([0])
            for point in range(len(X_train)):
                point_dist = minkowski(point, row)
                np.append(distance, point_dist)
            nearest_neighbors = np.argsort(distance)[:a]
            test_labels = y[nearest_neighbors]
            nearest = mode(test_labels)
            nearest = nearest.mode[0]
            labels.append(nearest)
        return labels


data = pd.read_csv(
    r'testData.csv')
df = pd.DataFrame(data)
df['artists'] = df['artists'].apply(
    lambda x: x.strip("[\"]").replace("'", "").split(',')[0])
df = df.drop(columns=['artists', 'name']).round(decimals=4)
correlation = df.corr(method='pearson').abs()
flatten = correlation.unstack().sort_values(
    ascending=False, kind='quicksort').dropna()


# We do not want the year to influence the data
# So we split our data into month, day and weekday
# Weekday is not clear to me but its just and index that starts
# on Monday = 0 and Sunday = 6
df['release_date'] = pd.to_datetime(df['release_date'])
df['month_release'] = df['release_date'].apply(lambda x: x.month)
df['day_release'] = df['release_date'].apply(lambda x: x.day)
df['weekday_release'] = df['release_date'].apply(lambda x: x.weekday())
df = df.drop(columns=['release_date'])
days = [x for x in df['day_release']]
# We need to assumed popularity so we drop the popularity to test the data
scale = MinMaxScaler(feature_range=(-1, 1))
X_train = df.sample(n=500).drop(columns='popularity').to_numpy()
X_test = df.sample(n=500).drop(columns='popularity').to_numpy()
y_train = df['popularity'].sample(n=500).to_numpy()
y_test = df['popularity'].sample(n=500).to_numpy()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.fit_transform(X_test)
knn = KNN()
prediction = knn.predict(X_train=X_train, X_test=X_test, y=y_train, a=12)
print(accuracy_score(y_test, prediction))
model = KNeighborsClassifier(n_neighbors=5, weights='distance')
model.fit(X_train, y_train)
predict = model.predict(X_test)
print(accuracy_score(y_test, predict))

from numpy import e, dot, log
from numpy.random import rand


class logistic_reg:
    def sigmoid(self, z):
        return 1/(1 + e**(-z))

    def cost(self, X, y, weights):
        z = dot(X, weights)
        prediction_one = y * log(self.sigmoid(z))
        prediction_two = (1 - y) * log(1 - self.sigmoid(z))
        return -sum(prediction_one + prediction_two) / len(X)

    def fit(self, X, y, epochs=25, lr=0.05):
        loss = []
        weights = rand(X.shape[1])
        N = len(X)
        # Gradient Descent Portion
        for _ in range(epochs):
            yhat = self.sigmoid(dot(X, weights))
            weights -= lr * dot(X.T, yhat - y) / N
            loss.append(self.cost(X, y, weights))

        self.weights = weights
        self.loss = loss

    def predict(self, X):
        z = dot(X, self.weights)
        # This will return a Binary result
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]

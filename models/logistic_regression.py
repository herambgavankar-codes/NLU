import math
from utils.logger import log

class LogisticRegression:
    def __init__(self, lr=0.05, epochs=20):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        if z < -20: return 0
        if z > 20: return 1
        return 1 / (1 + math.exp(-z))

    def fit(self, X, y):
        log("Starting Logistic Regression training")
        self.weights = {}

        for epoch in range(self.epochs):
            if epoch % 5 == 0:
                log(f"Epoch {epoch}")

            for xi, yi in zip(X, y):
                z = sum(self.weights.get(i,0)*v for i,v in xi.items())
                pred = self.sigmoid(z)
                error = pred - yi

                for i,v in xi.items():
                    self.weights[i] = self.weights.get(i,0) - self.lr * error * v

        log("Logistic Regression training done")

    def predict(self, X):
        preds = []
        for xi in X:
            z = sum(self.weights.get(i,0)*v for i,v in xi.items())
            preds.append(1 if self.sigmoid(z)>=0.5 else 0)
        return preds

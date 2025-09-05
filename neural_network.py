import numpy as np
import scipy


# Iteration (W,b) and loop to X 
class PerceptronSoftmax:
    def __init__(self, learning_rate=0.1, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def initialize(self, X, n_classes):
        np.random.seed(42)
        self.W = np.random.randn(X.shape[1], n_classes)
        self.b = np.random.randn(1, n_classes)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def model(self, X):
        Z = np.dot(X, self.W) + self.b
        return self.softmax(Z)

    def log_loss(self, A, y):
        m = len(y)
        Y = np.zeros_like(A)
        Y[np.arange(m), y] = 1

        # Adjust the weight
        class_counts = np.bincount(y)
        weights = 1.0 / (class_counts + 1e-10)
        sample_weights = weights[y]

        losses = -np.sum(Y * np.log(A + 1e-15), axis=1)
        weighted_loss = np.sum(sample_weights * losses) / m
        return weighted_loss

    def gradients(self, X, A, y):
        m = len(y)
        Y = np.zeros_like(A)
        Y[np.arange(m), y] = 1
        dW = np.dot(X.T, (A - Y)) / m
        db = np.sum(A - Y, axis=0, keepdims=True) / m
        return dW, db

    def update(self, dW, db):
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        self.initialize(X, n_classes)
        self.losses = []

        for iter in range(self.n_iter):
            A = self.model(X)
            loss = self.log_loss(A, y)
            self.losses.append(loss)
            dW, db = self.gradients(X, A, y)
            self.update(dW, db)

    def predict(self, X):
        A = self.model(X)
        return np.argmax(A, axis=1)

    def predict_proba(self, X):
        return self.model(X)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

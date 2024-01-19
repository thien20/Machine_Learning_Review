import numpy as np
import matplotlib.pyplot as plt

class RegularizedLinear:
    def __init__(self, learning_rate, ld):
        self.w = None
        self.b = None
        self.cost_his = []
        self.learning_rate = learning_rate
        self.ld = ld

    def fit_linear(self, x):
        return np.dot(x, self.w) + self.b
    
    def fit(self, X, y, epochs):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0
        self.cost_his = []
        
        for i in range(epochs):
            y_hat = self.fit_linear(X)
            d_error = y_hat - y

            gradient = np.dot(X.T, d_error) / num_samples + self.ld * self.w / num_samples
            self.w = self.w - self.learning_rate * gradient
            self.b = self.b - self.learning_rate * np.mean(d_error)

            cost = np.sum(d_error ** 2) / (2 * num_samples) + self.ld * np.sum(self.w ** 2) / (2 * num_samples)
            self.cost_his.append(cost)
        return self.w, self.b
    
    def predict(self, X):
        return self.fit_linear(X)

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.title("Cost function")
        plt.show()


if __name__ == "__main__":
    X = np.array([[100, 1], [100, 2], [100, 3], [100, 4]])
    X_normalized = X / np.max(X, axis=0)
    print("This is X_normalized: ", X_normalized)

    y = np.array([200, 300, 400, 500])
    X_test = np.array([[100, 5]])

    # Reduced learning rate and increased iterations
    gd = RegularizedLinear(learning_rate=0.1, ld=0.1)
    gd.fit(X_normalized, y, epochs=10000)
    gd.plot_cost()
    print("This is w_trained: {} - b_trained: {}".format(gd.w, gd.b))
    print("Prediction: ", gd.predict(X_test/np.max(X_test, axis=0)))
    

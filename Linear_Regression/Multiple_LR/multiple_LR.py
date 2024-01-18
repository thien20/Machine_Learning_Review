import numpy as np
import matplotlib.pyplot as plt

class MultipleGrad:
    def __init__(self, lr, iterations):
        self.lr = lr
        self.iterations = iterations
        self.w = None
        self.cost_history = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.cost_history = []

        for i in range(self.iterations):
            y_current = np.dot(X, self.w)
            error = (y_current - y)
            gradient = np.dot(X.T, error) / num_samples
            self.w -= self.lr * gradient
            cost = np.sum(error ** 2) / (2 * num_samples)
            self.cost_history.append(cost)
    
    def plot_cost_history(self):
        plt.plot(range(self.iterations), self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Gradient Descent - Cost History')
        plt.show()

    def predict(self, X_test):
        return np.dot(X_test, self.w)
    
if __name__ == "__main__":
    """
    note: the normalized technique is only used for Linear Regression
    """
    # size, number of bedrooms, number of floors

    X = np.array([[100, 4, 1], [200, 4, 2], [250, 5, 2], [130, 3, 2]])
    X_normalized = X / np.max(X, axis=0)
    y = np.array([1000, 2000, 3000, 1500])

    
    x_test = np.array([[300, 6, 3]])/np.max(X, axis=0)

    mgd = MultipleGrad(lr=0.1, iterations=10000)
    mgd.fit(X_normalized, y)
    mgd.plot_cost_history()
    print("trained weights: ", mgd.w/np.max(X, axis=0))
    print("prediction: ", mgd.predict(x_test))
    
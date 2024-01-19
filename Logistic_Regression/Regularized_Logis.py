import numpy as np
import matplotlib.pyplot as plt

class RegularizedLogis:
    def __init__(self, learning_rate, ld):
        self.w = None
        self.b = None
        self.cost_his = []
        self.learning_rate = learning_rate
        self.ld = ld

    def fit_sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, epochs):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0
        self.cost_his = []
        
        for i in range(epochs):
            # compute the error ~ loss 
            # --> gradient descent wrt w and b from the derivative of the cost function
            y_hat = self.fit_sigmoid(np.dot(X, self.w) + self.b)
            d_error = y_hat - y
            gradient = np.dot(X.T, d_error) / num_samples + self.ld * self.w / num_samples
            self.w = self.w - self.learning_rate * gradient
            self.b = self.b - self.learning_rate * np.mean(d_error)
            # cost function of logistic regression
            cost = d_error / num_samples
            self.cost_his.append(cost)
        return self.w, self.b
    
    def predict(self, X):
        return self.fit_sigmoid(np.dot(X, self.w) + self.b)
    
    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.xlabel("Epochs")
        plt.ylabel("Cost")
        plt.title("Cost function")
        plt.show()


if __name__ == "__main__":
    X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    # generate labels
    y = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([[1, 1.5]])
    log_reg = RegularizedLogis(learning_rate=0.01, ld=0.1)
    log_reg.fit(X, y, epochs=10000)
    log_reg.plot_cost()
    print("This is w_trained: {} - b_trained: {}".format(log_reg.w, log_reg.b))
    pred = log_reg.predict(X_test)
    if pred >= 0.5:
        print("Predicted label: 1 - probability: {}".format(pred))
    else:
        print("Predicted label: 0 - probability: {}".format(1-pred))
import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.cost_history = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        # initialize theta = 0 with respect to the function f(x) = theta_0 + theta_1 * x
        self.theta = np.zeros(num_features)
        self.cost_history = []
        
        for i in range(self.num_iterations):
            # calculate the current y values
            y_pred = np.dot(X, self.theta)

            error = (y_pred - y)
            # calculate the gradient with respect to 
            # the derivative of the cost function(theta_0 and theta_1)
            gradient = np.dot(X.T, error) / num_samples # can be understood as the average of the derivative of the cost function

            self.theta -= self.learning_rate * gradient
            
            cost = np.sum(error ** 2) / (2 * num_samples)
            self.cost_history.append(cost)
    
    def predict(self, X):
        return np.dot(X, self.theta)
    
    def plot_fit_and_pred(self, X, y):
        plt.scatter(X[:, 1], y)
        plt.plot(X[:, 1], self.predict(X), c='r')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Gradient Descent - Fit and Prediction')
        plt.show()
        

    def plot_cost_history(self):
        plt.plot(range(self.num_iterations), self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Gradient Descent - Cost History')
        plt.show()

if __name__ == "__main__":
    # Generate data and normalize features
    X = np.array([[100, 1], [100, 2], [100, 3], [100, 4]])
    X_normalized = X / np.max(X, axis=0)
    print("This is X_normalized: ", X_normalized)

    y = np.array([200, 300, 400, 500])

    # Reduced learning rate and increased iterations
    gd = GradientDescent(learning_rate=0.1, num_iterations=1000)
    gd.fit(X_normalized, y)
    gd.plot_cost_history()
    print("Learned Parameters (theta):", gd.theta)

    X_pred = np.array([[100, 5]]) / np.max(X, axis=0)
    prediction = gd.predict(X_pred)
    print("Prediction for X=[100, 5]:", prediction)

    X_normalized = np.append(X_normalized, X_pred).reshape(-1, 2)
    y = np.append(y, prediction)
    print("This is last X_normalized: ", X_normalized)
    gd.plot_fit_and_pred(X_normalized, y)

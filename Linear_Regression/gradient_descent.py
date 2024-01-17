import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, learning_rate=0.01, num_iterations=100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.cost_history = None
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.theta = np.zeros(num_features)
        self.cost_history = []
        
        for i in range(self.num_iterations):
            y_pred = np.dot(X, self.theta)
            error = (y_pred - y)

            gradient = np.dot(X.T, error) / num_samples

            self.theta -= self.learning_rate * gradient
            
            cost = np.sum(error ** 2) / (2 * num_samples)
            self.cost_history.append(cost)
    
    def predict(self, X):
        return np.dot(X, self.theta)
    
    def plot_cost_history(self):
        plt.plot(range(self.num_iterations), self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Gradient Descent - Cost History')
        plt.show()

# Geneate data
X = np.array([[100, 1], [100, 2], [100, 3], [100, 4]])
y = np.array([200, 300, 400, 500])

gd = GradientDescent(learning_rate=0.1, num_iterations=100)
gd.fit(X, y)
# gd.plot_cost_history()
print("prediction: ",gd.predict([[100, 5]]))

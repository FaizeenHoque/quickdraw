import cupy as cp

class nn:
    def __init__(self, xs, z1s, z2s, ys, learning_rate=0.01):
        self.xs = xs
        self.z1s = z1s
        self.z2s = z2s
        self.ys = ys
        self.learning_rate = learning_rate

        # Input to hidden layer 1
        self.W1 = cp.random.randn(self.z1s, self.xs) * 0.1
        self.b1 = cp.zeros((self.z1s, 1))

        # Hidden layer 1 to hidden layer 2
        self.W2 = cp.random.randn(self.z2s, self.z1s) * 0.1
        self.b2 = cp.zeros((self.z2s, 1))

        # Hidden layer 2 to output layer
        self.W3 = cp.random.randn(self.ys, self.z2s) * 0.1
        self.b3 = cp.zeros((self.ys, 1))

    # Load dataset
    def load_data(self, file_path):
        pass

    # Activation functions
    def relu(self, z):
        return cp.maximum(0, z)
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    def softmax(self, z):
        exp = cp.exp(z - cp.max(z, axis=0, keepdims=True))
        return exp / cp.sum(exp, axis=0, keepdims=True)

    # forward pass, loss calculation and backpropagation
    def forwardpass(self, x):
        # input to hidden layer 1
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.relu(self.z1)

        # hidden layer 1 to hidden layer 2
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.relu(self.z2)

        # hidden layer 2 to output layer
        self.z3 = self.W3 @ self.a2 + self.b3
        self.y_pred = self.softmax(self.z3)

        return self.y_pred
    def loss(self, y_pred, y_true):
        pass
    def backprop(self, x, y_true):
        pass

    # Training and prediction methods
    def train(self, X, y, epochs):
        pass
    def predict(self, X):
        pass

    # Model saving and loading methods
    def save_model(self, file_path):
        pass
    def load_model(self, file_path):
        pass

if __name__ == "__main__":
    model = nn(784, 128, 64, 345, 0.01)
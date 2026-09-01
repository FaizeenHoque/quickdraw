import cupy as cp

class nn:
    def __init__(self, xs, z1s, z2s, ys, learning_rate=0.01):
        self.xs = xs
        self.z1s = z1s
        self.z2s = z2s
        self.ys = ys
        self.learning_rate = learning_rate

        # Input to hidden layer 1
        self.W1 = cp.random.randn(self.z1s, self.xs) * cp.sqrt(2 / self.xs)
        self.b1 = cp.zeros((self.z1s, 1))

        # Hidden layer 1 to hidden layer 2
        self.W2 = cp.random.randn(self.z2s, self.z1s) * cp.sqrt(2 / self.z1s)
        self.b2 = cp.zeros((self.z2s, 1))

        # Hidden layer 2 to output layer
        self.W3 = cp.random.randn(self.ys, self.z2s) * cp.sqrt(2 / self.z2s)
        self.b3 = cp.zeros((self.ys, 1))

    # Load dataset
    def load_data(self, file_path):
        pass

    # Activation functions
    def relu(self, z):
        return cp.maximum(0, z)
    def relu_derivative(self, z):
        return (z > 0).astype(cp.float32)
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
        return -cp.mean(cp.sum(y_true * cp.log(y_pred + 1e-8), axis=0))
    def backprop(self, x, y_true):
        batch_size = x.shape[1]

        # Output layer gradients
        dz3 = self.y_pred - y_true

        dW3 = (dz3 @ self.a2.T) / batch_size
        db3 = cp.mean(dz3, axis=1, keepdims=True)

        # Hidden layer 2 gradients
        da2 = self.W3.T @ dz3

        dz2 = da2 * self.relu_derivative(self.z2)

        dW2 = (dz2 @ self.a1.T) / batch_size
        db2 = cp.mean(dz2, axis=1, keepdims=True)

        # Hidden layer 1 gradients
        da1 = self.W2.T @ dz2

        dz1 = da1 * self.relu_derivative(self.z1)

        dW1 = (dz1 @ x.T) / batch_size
        db1 = cp.mean(dz1, axis=1, keepdims=True)

        # Update weights and biases
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1


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
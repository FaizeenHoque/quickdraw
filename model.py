class nn:
    def __init__(self, x, z1, z2, y, learning_rate=0.01):
        self.x = x
        self.z1 = z1
        self.z2 = z2
        self.y = y
        self.learning_rate = learning_rate

    # Load dataset
    def load_data(self, file_path):
        pass

    # Activation functions
    def relu(self, z):
        pass
    def relu_derivative(self, z):
        pass
    def softmax(self, z):
        pass

    # forward pass, loss calculation and backpropagation
    def forwardpass(self, x):
        pass
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
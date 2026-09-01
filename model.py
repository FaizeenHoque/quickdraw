class nn:
    def __init__(self):
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
    def train(self, X, y, epochs, learning_rate):
        pass
    def predict(self, X):
        pass

if __name__ == "__main__":
    model = nn()
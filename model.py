import numpy as np
import cupy as cp

class nn:
    def __init__(self, xs, z1s, z2s, ys, learning_rate, momentum=0.95):
        self.xs = xs
        self.z1s = z1s
        self.z2s = z2s
        self.ys = ys
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Input to hidden layer 1
        self.W1 = cp.random.randn(self.z1s, self.xs) * cp.sqrt(2 / self.xs)
        self.b1 = cp.zeros((self.z1s, 1))
        self.vW1 = cp.zeros_like(self.W1)
        self.vb1 = cp.zeros_like(self.b1)

        # Hidden layer 1 to hidden layer 2
        self.W2 = cp.random.randn(self.z2s, self.z1s) * cp.sqrt(2 / self.z1s)
        self.b2 = cp.zeros((self.z2s, 1))
        self.vW2 = cp.zeros_like(self.W2)
        self.vb2 = cp.zeros_like(self.b2)

        # Hidden layer 2 to output layer
        self.W3 = cp.random.randn(self.ys, self.z2s) * cp.sqrt(2 / self.z2s)
        self.b3 = cp.zeros((self.ys, 1))
        self.vW3 = cp.zeros_like(self.W3)
        self.vb3 = cp.zeros_like(self.b3)

    # Load dataset
    def load_data(self, x_path, y_path, classes_path):
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        classes = np.load(classes_path)

        return X, y, classes

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

        # hidden layer 1 to hidden l2ayer 2
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.relu(self.z2)

        # hidden layer 2 to output layer
        self.z3 = self.W3 @ self.a2 + self.b3
        self.y_pred = self.softmax(self.z3)

        return self.y_pred
    def loss(self, y_pred, y_true):
        batch_size = y_true.shape[0]

        correct_probs = y_pred[y_true, cp.arange(batch_size)]

        return -cp.mean(cp.log(correct_probs + 1e-8))
    def backprop(self, x, y_true):
        batch_size = x.shape[1]

        # Output layer gradients
        dz3 = self.y_pred.copy()

        dz3[y_true, cp.arange(batch_size)] -= 1
        dz3 /= batch_size

        dW3 = dz3 @ self.a2.T
        db3 = cp.sum(dz3, axis=1, keepdims=True)

        # Hidden layer 2 gradients
        da2 = self.W3.T @ dz3

        dz2 = da2 * self.relu_derivative(self.z2)

        dW2 = dz2 @ self.a1.T
        db2 = cp.sum(dz2, axis=1, keepdims=True)

        # Hidden layer 1 gradients
        da1 = self.W2.T @ dz2

        dz1 = da1 * self.relu_derivative(self.z1)

        dW1 = dz1 @ x.T
        db1 = cp.sum(dz1, axis=1, keepdims=True)

        # Update weights and biases
        self.vW3 = self.momentum * self.vW3 + dW3
        self.vb3 = self.momentum * self.vb3 + db3
        self.W3 -= self.learning_rate * self.vW3
        self.b3 -= self.learning_rate * self.vb3

        self.vW2 = self.momentum * self.vW2 + dW2
        self.vb2 = self.momentum * self.vb2 + db2
        self.W2 -= self.learning_rate * self.vW2
        self.b2 -= self.learning_rate * self.vb2

        self.vW1 = self.momentum * self.vW1 + dW1
        self.vb1 = self.momentum * self.vb1 + db1
        self.W1 -= self.learning_rate * self.vW1
        self.b1 -= self.learning_rate * self.vb1


    # Training and prediction methods
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, decay_every=10, decay_factor=0.5):
        n = X_train.shape[0]
        num_batches = (n + batch_size - 1) // batch_size  # Ceiling division to get the number of batches

        for epoch in range(epochs): 

            # Shuffle the data indices for each epoch
            indices = np.random.permutation(n)

            total_loss = 0
            correct = 0
            total = 0

            for start in range(0, n, batch_size):

                # Get the batch indices
                end = min(start + batch_size, n)
                batch_indices = indices[start:end]

                # Get the batch data
                X_batch = cp.asarray(X_train[batch_indices].T, dtype=cp.float32)
                y_batch = cp.asarray(y_train[batch_indices], dtype=cp.int32)

                # Forward pass
                y_pred = self.forwardpass(X_batch)

                # Compute loss
                loss_value = self.loss(y_pred, y_batch)

                total_loss += loss_value * X_batch.shape[1]

                # Backpropagation
                self.backprop(X_batch, y_batch)

                # Compute accuracy
                predictions = cp.argmax(y_pred, axis=0)
                labels = y_batch

                # Update correct and total counts
                correct += cp.sum(predictions == labels)
                total += X_batch.shape[1]
                
                batch_num = start // batch_size + 1

                batch_loss = float(loss_value)
                batch_accuracy = float(cp.mean(predictions == y_batch))

                progress = batch_num / num_batches

                bar_length = 30
                filled = int(bar_length * progress)
                bar = "=" * filled + "-" * (bar_length - filled)

                print(
                    f"\rEpoch {epoch + 1}/{epochs} "
                    f"Batch {batch_num}/{num_batches} "
                    f"[{bar}] "
                    f"Loss: {batch_loss:.4f} "
                    f"Accuracy: {batch_accuracy * 100:.2f}%",
                    end="",
                    flush=True
                )
                
            if (epoch + 1) % decay_every == 0:
                self.learning_rate *= decay_factor
                print(f"Learning rate decayed to {self.learning_rate:.6f}")

            # Compute average loss and accuracy for the epoch
            average_loss = float(total_loss / n)
            accuracy = float(correct / total)
            
            val_loss, val_accuracy = self.evaluate(X_val, y_val, batch_size)

            # Print epoch statistics
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {average_loss:.4f} "
                f"- Accuracy: {accuracy * 100:.2f}%"
                f"- Val Loss: {val_loss:.4f} "
                f"- Val Accuracy: {val_accuracy * 100:.2f}%"
            )

    def predict(self, X):
        X = cp.asarray(X, dtype=cp.float32).T
        y_pred = self.forwardpass(X)
        predictions = cp.argmax(y_pred, axis=0)
        return cp.asnumpy(predictions)
    
    def evaluate(self, X, y, batch_size=512):
        n = X.shape[0]
        total_loss = 0
        correct = 0
        total = 0

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = cp.asarray(X[start:end].T, dtype=cp.float32)
            y_batch = cp.asarray(y[start:end], dtype=cp.int32)

            y_pred = self.forwardpass(X_batch)
            loss_value = self.loss(y_pred, y_batch)
            
            batch_count = X_batch.shape[1]

            total_loss += loss_value * batch_count

            predictions = cp.argmax(y_pred, axis=0)
        
            correct += cp.sum(predictions == y_batch).get()
            total += batch_count

        return total_loss / total, correct / total
    

    # Model saving and loading methods
    def save_model(self, file_path):
        np.savez(
            file_path,
            W1=cp.asnumpy(self.W1),
            b1=cp.asnumpy(self.b1),
            W2=cp.asnumpy(self.W2),
            b2=cp.asnumpy(self.b2),
            W3=cp.asnumpy(self.W3),
            b3=cp.asnumpy(self.b3),
            vW1=cp.asnumpy(self.vW1),
            vb1=cp.asnumpy(self.vb1),
            vW2=cp.asnumpy(self.vW2),
            vb2=cp.asnumpy(self.vb2),
            vW3=cp.asnumpy(self.vW3),
            vb3=cp.asnumpy(self.vb3),
            xs=self.xs,
            z1s=self.z1s,
            z2s=self.z2s,
            ys=self.ys,
            learning_rate=self.learning_rate
        )
    def load_model(self, file_path):
        data = np.load(file_path)
        self.W1 = cp.asarray(data['W1'])
        self.b1 = cp.asarray(data['b1'])
        self.W2 = cp.asarray(data['W2'])
        self.b2 = cp.asarray(data['b2'])
        self.W3 = cp.asarray(data['W3'])
        self.b3 = cp.asarray(data['b3'])

        self.vb1 = cp.asarray(data['vb1'])
        self.vW1 = cp.asarray(data['vW1'])
        self.vb2 = cp.asarray(data['vb2'])
        self.vW2 = cp.asarray(data['vW2'])
        self.vb3 = cp.asarray(data['vb3'])
        self.vW3 = cp.asarray(data['vW3'])

        self.xs = int(data['xs'])
        self.z1s = int(data['z1s'])
        self.z2s = int(data['z2s'])
        self.ys = int(data['ys'])
        self.learning_rate = float(data['learning_rate'])

if __name__ == "__main__":
    model = nn(784, 1024, 512, 344, 0.01)
    model.load_model("quickdraw_model.npz")
    model.learning_rate = 0.005
    print(f"Loaded architecture: {model.xs} -> {model.z1s} -> {model.z2s} -> {model.ys}")
    
    X_train, y_train, classes_train = model.load_data("quickdraw_X_train.npy", "quickdraw_y_train.npy", "quickdraw_classes.npy")
    X_val, y_val, classes_val = model.load_data("quickdraw_X_val.npy", "quickdraw_y_val.npy", "quickdraw_classes.npy")
    
    model.train(X_train, y_train, X_val, y_val, epochs=1, batch_size=4096, decay_every=1, decay_factor=0.998)
    model.save_model("quickdraw_model.npz")

    # idx = np.random.choice(len(X), 10, replace=False)
    # prediction = model.predict(X[idx])

    # predicted_labels = classes[prediction]
    # true_labels = classes[y[idx]]

    # for pred, true in zip(predicted_labels, true_labels):
    # 	print(f"Predicted: {pred:20s} | Actual: {true}")
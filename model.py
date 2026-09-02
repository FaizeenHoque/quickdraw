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
        self.vW3 = self.momentum * self.vW3 + (1 - self.momentum) * dW3
        self.vb3 = self.momentum * self.vb3 + (1 - self.momentum) * db3
        self.W3 -= self.learning_rate * self.vW3
        self.b3 -= self.learning_rate * self.vb3

        self.vW2 = self.momentum * self.vW2 + (1 - self.momentum) * dW2
        self.vb2 = self.momentum * self.vb2 + (1 - self.momentum) * db2
        self.W2 -= self.learning_rate * self.vW2
        self.b2 -= self.learning_rate * self.vb2

        self.vW1 = self.momentum * self.vW1 + (1 - self.momentum) * dW1
        self.vb1 = self.momentum * self.vb1 + (1 - self.momentum) * db1
        self.W1 -= self.learning_rate * self.vW1
        self.b1 -= self.learning_rate * self.vb1

    # Training and prediction methods
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, chunk_size=200_000, decay_every=10, decay_factor=0.5, checkpoint_every=10, checkpoint_path="quickdraw_checkpoint"):
        n = X_train.shape[0]
        num_batches = (n + batch_size - 1) // batch_size

        for epoch in range(epochs):
            chunk_starts = list(range(0, n, chunk_size))
            np.random.shuffle(chunk_starts)

            total_loss = 0
            correct = 0
            total = 0
            batch_num = 0

            for chunk_start in chunk_starts:
                chunk_end = min(chunk_start + chunk_size, n)

                X_chunk = np.asarray(X_train[chunk_start:chunk_end])
                y_chunk = np.asarray(y_train[chunk_start:chunk_end])

                chunk_perm = np.random.permutation(len(X_chunk))
                X_chunk = X_chunk[chunk_perm]
                y_chunk = y_chunk[chunk_perm]

                for start in range(0, len(X_chunk), batch_size):
                    end = min(start + batch_size, len(X_chunk))

                    X_batch = cp.asarray(X_chunk[start:end].T, dtype=cp.float32)
                    y_batch = cp.asarray(y_chunk[start:end], dtype=cp.int32)

                    y_pred = self.forwardpass(X_batch)
                    loss_value = self.loss(y_pred, y_batch)
                    total_loss += loss_value * X_batch.shape[1]

                    self.backprop(X_batch, y_batch)

                    predictions = cp.argmax(y_pred, axis=0)
                    correct += cp.sum(predictions == y_batch)
                    total += X_batch.shape[1]
                    batch_num += 1

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

            print()

            if (epoch + 1) % decay_every == 0:
                self.learning_rate *= decay_factor
                print(f"Learning rate decayed to {self.learning_rate:.6f}")

            average_loss = float(total_loss / n)
            accuracy = float(correct / total)
            val_loss, val_accuracy = self.evaluate(X_val, y_val, batch_size)

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"- Loss: {average_loss:.4f} "
                f"- Accuracy: {accuracy * 100:.2f}%"
                f"- Val Loss: {val_loss:.4f} "
                f"- Val Accuracy: {val_accuracy * 100:.2f}%"
            )
            
            if (epoch + 1) % checkpoint_every == 0:
                path = f"{checkpoint_path}_epoch{epoch + 1}.npz"
                self.save_model(path)
                print(f"Checkpoint saved: {path}")
        
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
    
    model = nn(784, 1024, 512, 344, 0.02)
    # model.load_model("quickdraw_model.npz")
    # model.learning_rate = 0.005
    # print(f"Loaded architecture: {model.xs} -> {model.z1s} -> {model.z2s} -> {model.ys}")
    
    X_train, y_train, classes_train = model.load_data("quickdraw_X_train.npy", "quickdraw_y_train.npy", "quickdraw_classes.npy")
    X_val, y_val, classes_val = model.load_data("quickdraw_X_val.npy", "quickdraw_y_val.npy", "quickdraw_classes.npy")

    model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=1024, decay_every=10, decay_factor=0.7, checkpoint_every=10)
    
    all_preds = []
    for start in range(0, X_val.shape[0], 2048):
        end = min(start + 2048, X_val.shape[0])
        xb = cp.asarray(X_val[start:end].T, dtype=cp.float32)
        pred = model.forwardpass(xb)
        all_preds.append(cp.asnumpy(cp.argmax(pred, axis=0)))

    all_preds = np.concatenate(all_preds)
    unique, counts = np.unique(all_preds, return_counts=True)
    print("number of distinct classes predicted:", len(unique))
    print("top 10 most-predicted classes:", unique[np.argsort(-counts)[:10]], "with counts:", np.sort(counts)[::-1][:10])

    for name, param in [("W1", model.W1), ("W2", model.W2), ("W3", model.W3)]:
        print(name, "max:", float(cp.max(cp.abs(param))), "has nan:", bool(cp.any(cp.isnan(param))))

    model.save_model("quickdraw_model.npz")

    # idx = np.random.choice(len(X), 10, replace=False)
    # prediction = model.predict(X[idx])

    # predicted_labels = classes[prediction]
    # true_labels = classes[y[idx]]

    # for pred, true in zip(predicted_labels, true_labels):
    # 	print(f"Predicted: {pred:20s} | Actual: {true}")
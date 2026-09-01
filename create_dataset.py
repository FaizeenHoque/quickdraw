import numpy as np
import pathlib
import matplotlib.pyplot as plt
import time

X = []
y = []

classes = sorted(pathlib.Path("data").glob("*.npy"))

for label, file in enumerate(classes):
	print(f"Loading category: {file.stem}...")

	data = np.load(file)
	data = data[:10000]  # Limit to 10000 samples per class

	# Normalize pixels from 0-255 to 0-1
	data = data.astype(np.float32) / 255.0

	X.append(data)
	y.extend([label] * len(data))

# turn lists into numpy arrays
t0 = time.time()
X = np.concatenate(X, axis=0)
y = np.array(y)
print(f"Concatenate took {time.time() - t0:.1f}s")

# shuffle X and y together (same permutation) so the saved file
# isn't grouped by class, useful if training later reads sequential slices
t0 = time.time()
shuffle_order = np.random.permutation(len(X))
X = X[shuffle_order]
y = y[shuffle_order]
print(f"Shuffle took {time.time() - t0:.1f}s")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of classes:", len(classes))

t0 = time.time()
np.savez(
	"quickdraw_dataset.npz",
	X=X,
	y=y,
	classes=np.array([file.stem for file in classes])
)
print(f"Save took {time.time() - t0:.1f}s")
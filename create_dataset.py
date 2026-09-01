import numpy as np
import pathlib
import matplotlib.pyplot as plt

X = []
y = []

classes = sorted(pathlib.Path("data").glob("*.npy"))

for label, file in enumerate(classes):
    print(f"Loading category: {file.stem}...")

    data = np.load(file)
    data = data[:5000]  # Limit to 5000 samples per class

    # Normalize pixels from 0-255 to 0-1
    data = data.astype(np.float32) / 255.0

    X.append(data)
    y.extend([label] * len(data))

# turn lists into numpy arrays
X = np.concatenate(X, axis=0)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of classes:", len(classes))

np.savez_compressed(
    "quickdraw_dataset.npz",
    X=X,
    y=y,
    classes=np.array([file.stem for file in classes])
)
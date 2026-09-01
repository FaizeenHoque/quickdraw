import numpy as np
import pathlib
import time

DATA_DIR = pathlib.Path("data")
SAMPLES_PER_CLASS = 20_000

files = sorted(DATA_DIR.glob("*.npy"))

if not files:
    raise RuntimeError("No .npy files found in data/")

classes = np.array([file.stem for file in files])

num_classes = len(files)
total_samples = num_classes * SAMPLES_PER_CLASS

print("Classes:", num_classes)
print("Total samples:", total_samples)

print("Creating disk-backed arrays...")

X = np.lib.format.open_memmap(
    "quickdraw_X.npy", mode="w+", dtype=np.float32, shape=(total_samples, 784)
)

y = np.lib.format.open_memmap(
    "quickdraw_y.npy", mode="w+", dtype=np.int32, shape=(total_samples,)
)

offset = 0

for label, file in enumerate(files):

    print(f"[{label + 1}/{num_classes}] " f"Loading {file.stem}...")

    # mmap the source too
    data = np.load(file, mmap_mode="r")
    count = min(SAMPLES_PER_CLASS, len(data))

    # Only this class's samples are processed
    chunk = np.asarray(data[:count], dtype=np.float32)
    chunk /= 255.0

    X[offset : offset + count] = chunk
    y[offset : offset + count] = label

    offset += count

    del chunk
    del data

print("Flushing to disk...")

X.flush()
y.flush()

del X
del y

np.save("quickdraw_classes.npy", classes)

print()
print("Done!")
print("X:", "quickdraw_X.npy")
print("y:", "quickdraw_y.npy")
print("classes:", "quickdraw_classes.npy")

import numpy as np
import pathlib

DATA_DIR = pathlib.Path("data")

SAMPLES_PER_CLASS = 12_000
TRAIN_RATIO = 0.9

files = sorted(DATA_DIR.glob("*.npy"))
if not files:
    raise RuntimeError("No .npy files found in data/")
if len(files) != 344:
    raise RuntimeError(f"Expected 344 classes, found {len(files)}")

classes = np.array([file.stem for file in files])

num_classes = len(files)

train_per_class = int(SAMPLES_PER_CLASS * TRAIN_RATIO)
val_per_class = SAMPLES_PER_CLASS - train_per_class

total_train = num_classes * train_per_class
total_val = num_classes * val_per_class

print("Classes:", num_classes)
print("Train samples:", total_train)
print("Validation samples:", total_val)
print("Total samples:", total_train + total_val)

print()
print("Creating disk-backed arrays...")
pathlib.Path("dataset").mkdir(exist_ok=True)

X_train = np.lib.format.open_memmap(
    "dataset/quickdraw_X_train.npy", mode="w+", dtype=np.float32, shape=(total_train, 784)
)

y_train = np.lib.format.open_memmap(
    "dataset/quickdraw_y_train.npy", mode="w+", dtype=np.int32, shape=(total_train,)
)

X_val = np.lib.format.open_memmap(
    "dataset/quickdraw_X_val.npy", mode="w+", dtype=np.float32, shape=(total_val, 784)
)

y_val = np.lib.format.open_memmap(
    "dataset/quickdraw_y_val.npy", mode="w+", dtype=np.int32, shape=(total_val,)
)

train_offset = 0
val_offset = 0

rng = np.random.default_rng(42)

for label, file in enumerate(files):

    print(f"[{label + 1}/{num_classes}] " f"Loading {file.stem}...")

    # Memory-map source dataset
    data = np.load(file, mmap_mode="r")

    count = min(SAMPLES_PER_CLASS, len(data))
    
    if count < SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"{file.stem} has only {len(data)} samples, "
            f"expected at least {SAMPLES_PER_CLASS}"
        )

    # Randomize this class before splitting
    indices = rng.permutation(count)
    
    train_indices = indices[:train_per_class]
    val_indices = indices[train_per_class:]

    # Load only the required samples
    train_chunk = np.asarray(data[train_indices], dtype=np.float32)

    val_chunk = np.asarray(data[val_indices], dtype=np.float32)

    # Normalize 0-255 → 0-1
    train_chunk /= 255.0
    val_chunk /= 255.0

    # Write directly to disk-backed arrays
    X_train[train_offset : train_offset + len(train_chunk)] = train_chunk

    y_train[train_offset : train_offset + len(train_chunk)] = label

    X_val[val_offset : val_offset + len(val_chunk)] = val_chunk

    y_val[val_offset : val_offset + len(val_chunk)] = label

    train_offset += len(train_chunk)
    val_offset += len(val_chunk)

    del train_chunk
    del val_chunk
    del indices
    del train_indices
    del val_indices
    del data

print()
print("Flushing to disk...")

X_train.flush()
y_train.flush()
X_val.flush()
y_val.flush()

del X_train
del y_train
del X_val
del y_val

np.save("dataset/quickdraw_classes.npy", classes)

print()
print("Done!")
print()
print("Training:")
print("  quickdraw_X_train.npy")
print("  quickdraw_y_train.npy")
print()
print("Validation:")
print("  quickdraw_X_val.npy")
print("  quickdraw_y_val.npy")
print()
print("Classes:")
print("  quickdraw_classes.npy")

import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader

class QuickDrawDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X_path = X_path
        self.y_path = y_path
        self.X = None
        self.y = None

    def _lazy_init(self):
        if self.X is None:
            self.X = np.load(self.X_path, mmap_mode="r")
            self.y = np.load(self.y_path, mmap_mode="r")

    def __len__(self):
        self._lazy_init()
        return len(self.X)

    def __getitem__(self, idx):
        self._lazy_init()
        return self.X[idx], self.y[idx]

def collate_fn(batch):
    X, y = zip(*batch)

    X = torch.from_numpy(np.asarray(X)).float()
    y = torch.from_numpy(np.asarray(y)).long()

    X = X.reshape(-1, 1, 28, 28)

    return X, y

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2
        )
        self.fc1 = nn.Linear(
            in_features=128 * 3 * 3,
            out_features=768
        )
        self.fc2 = nn.Linear(
            in_features=768,
            out_features=344
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
        
        
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("fork", force=True)

    train_dataset = QuickDrawDataset("dataset/quickdraw_X_train.npy", "dataset/quickdraw_y_train.npy")
    val_dataset = QuickDrawDataset("dataset/quickdraw_X_val.npy", "dataset/quickdraw_y_val.npy")
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (f"Using device: {device}")
    
    model = Model().to(device)
    
    
    model.load_state_dict(
        torch.load(
            "quickdraw_model.pth",
            map_location=device
        )
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )
    
    model.train()
    print("Starting training...")
    
    best_val_accuracy = 71.06
    for epoch in range(100):
        model.train()
        
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

            if batch_idx % 100 == 0:
                predictions = outputs.argmax(dim=1)
                accuracy = (predictions == y).float().mean()

                print(
                    f"Epoch {epoch + 1} "
                    f"Batch {batch_idx + 1}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f} "
                    f"Accuracy: {accuracy.item() * 100:.2f}%",
                    end="\r",
                    flush=True
                )
                
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = correct / total * 100
        
        print(
            f"Epoch {epoch + 1}/100 "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.2f}%"
        )
        
        model.eval()
        
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == y_val).sum().item()
                val_total += y_val.size(0)
        
        val_accuracy = val_correct / val_total * 100
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                "quickdraw_model.pth"
            )

            print(f"New best model saved! ({val_accuracy:.2f}%)")
    
        scheduler.step()
        
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
            
    
        
            
        


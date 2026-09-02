import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader

class QuickDrawDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(
            self.X[idx],
            dtype=torch.float32
        ).reshape(1, 28, 28)  
        
        y = torch.tensor(
            self.y[idx],
            dtype=torch.long
        )
        return x, y

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2
        )
        self.fc1 = nn.Linear(
            in_features=16 * 7 * 7,
            out_features=128
        )
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=344
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
        
        
if __name__ == "__main__":
    X_train = np.load("quickdraw_X_train.npy", mmap_mode="r")
    y_train = np.load("quickdraw_y_train.npy", mmap_mode="r")

    X_val = np.load("quickdraw_X_val.npy", mmap_mode="r")
    y_val = np.load("quickdraw_y_val.npy", mmap_mode="r")
    
    train_dataset = QuickDrawDataset(X_train, y_train)
    val_dataset = QuickDrawDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (f"Using device: {device}")
    
    model = Model().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    print("Starting training...")
    
    for epoch in range(10):
        model.train()
        
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
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
            f"Epoch {epoch + 1}/10 "
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
        
        
            
        


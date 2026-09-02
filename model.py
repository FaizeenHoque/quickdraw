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
        x = torch.tesnor(
            self.X[idx],
            dtype=torch.float32
        ).respape(1, 28, 28)  
        
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
    X_train = np.load("quickdraw_X_train.npy")        
    y_train = np.load("quickdraw_y_train.npy")
    
    X_val = np.load("quickdraw_X_val.npy")
    y_val = np.load("quickdraw_y_val.npy")


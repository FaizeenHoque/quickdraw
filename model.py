from torch import nn

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
        
        
        

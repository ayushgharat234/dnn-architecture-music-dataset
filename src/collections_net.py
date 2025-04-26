# Importing necessary libraries and dependencies
import torch 
import torch.nn as nn
import torch.nn.functional as F

# Creating custom Neural Networks for experimentation

# Model A: Basic Architecture: 2 hidden-layers
class ModelA(nn.Module):
    def __init__(self, input_dim=90, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
# Model B: Deeper Network
class ModelB(nn.Module):
    def __init__(self, input_dim=90, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
# Model C: Batch Normalization + Dropout
class ModelC(nn.Module):
    def __init__(self, input_dim=90, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),  # Testing with 0.5 though a hyperparameter
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
# Model D: Deep Narrow Network with SeLU
class ModelD(nn.Module):
    def __init__(self, input_dim=90, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
# Model E: Wide Deep Network with Swish
class ModelE(nn.Module):
    def __init__(self, input_dim=90, output_dim=10):
        super().__init__()
        self.swish = lambda x: x * torch.sigmoid(x) # Swish Activation: not in defaults
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.net[0](x)
        x = self.swish(x)
        x = self.net[1](x)
        x = self.swish(x)
        x = self.net[2](x)
        x = self.swish(x)
        x = self.net[3](x)
        return x

# Model F: Small Deep Network with LeakyReLU
class ModelF(nn.Module):
    def __init__(self, input_dim=90, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Model G: Small Simple Network with Tanh
class ModelG(nn.Module):
    def __init__(self, input_dim=90, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x)
    
# Model H: Very Deep Network
class ModelH(nn.Module):
    def __init__(self, input_dim=90, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, output_dim)
        )
    def forward(self, x):
        return self.net(x)
    
# Model I: GeLU Activation based network
class ModelI(nn.Module):
    def __init__(self, input_dim=90, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Model J: Extra-Wide Deep Network
class ModelJ(nn.Module):
    def __init__(self, input_dim=90, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)
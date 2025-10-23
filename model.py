import torch
from torch import nn # Neural network module
from torchvision import datasets # Datasets module, includes popular datasets like MNIST
from torchvision.transforms import ToTensor #to convert images to tensors   
from torch.utils.data import DataLoader # DataLoader module to handle batching and shuffling of data

train_data = datasets.MNIST(root='data', #where to store the data 
                            train=True, #get training data
                            download=True, #download if not already present
                            transform=ToTensor()) #convert images to tensors

test_data = datasets.MNIST(root='data', 
                            train=False, #get test dataset this time
                            download=True, 
                            transform=ToTensor()) 

# Create DataLoaders for batching and shuffling
batch_size = 64 #set batch size
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# Define the neural network model
class MyFirstClassifier(nn.Module):
    def __init__(self): # Initialize the model
        super(MyFirstClassifier, self).__init__() # Call the parent class constructor to initialize the nn.Module

        self.flatten = nn.Flatten() # Flatten layer to convert 2D tensors to 1D vectors

        self.linear_relu_stack = nn.Sequential( # Sequential container for layers
            nn.Linear(28*28, 512), # First linear layer
            nn.ReLU(),             # Rectified Linear Unit activation, basically adds non-linearity by zeroing out negative values
            nn.Linear(512, 512),   # Second linear layer
            nn.ReLU(),             # Another ReLU activation
            nn.Linear(512, 10)     # Output layer for 10 classes (0-9 digits)
        )

    def forward(self, x): # Define the forward pass of the model
        x = self.flatten(x) # Flatten the input
        logits = self.linear_relu_stack(x) # Pass through the sequential layers defined above
        
        return logits
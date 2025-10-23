import torch
from torch import nn
from model import MyFirstClassifier, train_loader

model = MyFirstClassifier() # Instantiate the model

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss() # Cross-entropy loss for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam optimizer with learning rate of 0.001 

# Training loop
num_epochs = 3 # Number of epochs to train the model

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    for batch, (images, labels) in enumerate(train_loader):
        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        
        # Compute the loss
        loss = loss_fn(outputs, labels)
        
        # Zero the gradients before running the backward pass
        optimizer.zero_grad()
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        optimizer.step()

        if batch % 100 == 0:
            print(f"  Batch {batch}, Loss: {loss.item():.4f}") # Print loss every 100 batches
print("Training complete.")

# Save the trained model
torch.save(model.state_dict(), "my_first_classifier.pth")
print("Model saved to my_first_classifier.pth")
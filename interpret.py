import torch
import matplotlib.pyplot as plt
from captum.attr import Saliency

from model import MyFirstClassifier, test_loader

#initialize the model
model = MyFirstClassifier()

#load the trained model weights
model.load_state_dict(torch.load("my_first_classifier.pth"))
model.eval()  # Set the model to evaluation mode as we are not training it here

print("Trained model loaded from my_first_classifier.pth")

images, labels = next(iter(test_loader))  # Get a batch of test images and labels
image, label = images[0], labels[0]  # Select the first image and label from the batch (change index for different images)
image = image.unsqueeze(0)  # Add batch dimension
image.requires_grad_()  # Enable gradient computation for the image for saliency calculation

print(f"Selected image label: {label.item()}")

# Initialize the Saliency object
saliency = Saliency(model)
# Compute saliency map
grads = saliency.attribute(image, target=label.item())

grads = torch.abs(grads)  # Take absolute value of gradients for better visualization

# Convert gradients to numpy for visualization
grads = grads.squeeze().detach().numpy() # Remove batch dimension, stop tracking gradients, convert to numpy
image = image.squeeze().detach().numpy() 

# Plot original image and saliency map
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image, cmap='gray')
ax[0].set_title(f'Original Image - Label: {label.item()}')
ax[0].axis('off')
ax[1].imshow(grads, cmap='hot')
ax[1].set_title('Saliency Map')
ax[1].axis('off')
#add color bar to saliency map
# cbar = plt.colorbar(ax[1].imshow(grads, cmap='hot'), ax=ax[1])
plt.savefig('saliency_map.png')
plt.show()



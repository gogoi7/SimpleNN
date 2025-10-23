import torch
import matplotlib.pyplot as plt
from captum.attr import Saliency, IntegratedGradients

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

#Initialize Integrated Gradients object
ig = IntegratedGradients(model)
#set baseline as a black image
baseline = torch.zeros_like(image)
# Compute Integrated Gradients
ig_attr = ig.attribute(image, baselines=baseline, target=label.item())
ig_attr = torch.abs(ig_attr)  

# Convert gradients to numpy for visualization
grads = grads.squeeze().detach().numpy() # Remove batch dimension, stop tracking gradients, convert to numpy
image = image.squeeze().detach().numpy() 
ig_attr = ig_attr.squeeze().detach().numpy()

# Plot original image, saliency map, and integrated gradients
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(grads, cmap='hot')
ax[1].set_title('Saliency Map')
ax[1].axis('off')
ax[2].imshow(ig_attr, cmap='hot')
ax[2].set_title('Integrated Gradients')
ax[2].axis('off')
plt.savefig('interpret_img.png')
plt.show()
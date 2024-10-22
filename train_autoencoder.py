import torch
from AutoDecoder import AutoDecoder
from utils import create_dataloaders, plot_tsne
from evaluate import evaluate_model
import matplotlib.pyplot as plt
import os

# Parameters
latent_dim = 128  # Increase latent dimension to 128
batch_size = 64
epochs = 20
learning_rate = 0.0005

# Load the dataset (train and test splits are handled in the function)
train_ds, train_dl, test_ds, test_dl = create_dataloaders(data_path='dataset', batch_size=batch_size)

# Debugging: Check if the datasets are loaded properly
print(f"Training dataset size: {len(train_ds)}")
print(f"Test dataset size: {len(test_ds)}")
print(f"Number of batches in training DataLoader: {len(train_dl)}")
print(f"Number of batches in test DataLoader: {len(test_dl)}")

# Initialize the CNN-based AutoDecoder model
model = AutoDecoder(latent_dim=latent_dim)

# Set up the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Initialize latent vectors for each sample in the training set
latent_vectors = torch.randn(len(train_ds), latent_dim, requires_grad=True, device=device)

# Optimizer for both model parameters and latent vectors
optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': latent_vectors}], lr=learning_rate, weight_decay=1e-4)

# Loss function: L1 Loss (Mean Absolute Error)
criterion = torch.nn.SmoothL1Loss()

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training Loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for idx, (i, data) in enumerate(train_dl):
        # Get the corresponding latent vectors for the batch
        latent_batch = latent_vectors[i].to(device)

        optimizer.zero_grad()

        # Forward pass using the latent vectors, not the image data
        reconstructions = model(latent_batch)

        # Remove the extra channel dimension to match the target size (batch_size, 28, 28)
        reconstructions = reconstructions.squeeze(1)

        # Compute loss by comparing the reconstructed image to the original image
        loss = criterion(reconstructions, data.float().to(device))

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_dl):.4f}")

    scheduler.step()

# Evaluate the model on the test set
model.eval()

# Create latent vectors for the test set and initialize them with random values
initial_latents = torch.randn(len(test_ds), latent_dim, device=device)

# Optimizer to optimize only the test set latent vectors (no weight decay needed here)
optimizer_test = torch.optim.Adam([initial_latents.requires_grad_()], lr=learning_rate)

# Evaluate the model without updating model parameters, only the latent vectors
test_loss = evaluate_model(model, test_dl, optimizer_test, initial_latents, epochs=10, device=device)
print(f"Test set evaluation loss: {test_loss:.4f}")

# Plot t-SNE visualization of the latent space after evaluation
plot_tsne(test_ds, initial_latents, file_name='tsne_plot.png', plot_title="t-SNE Visualization of Latents")


def plot_images(images, file_name, title):
    # Create the output directory if it doesn't exist
    os.makedirs('output_images', exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].cpu().detach().numpy(), cmap='gray')
        ax.axis('off')

    plt.suptitle(title)

    # Save the plot as a PNG file
    output_path = f"output_images/{file_name}.png"
    plt.savefig(output_path)
    print(f"Images saved to {output_path}")
    plt.close()


# 1. Sample 5 random latent vectors from U(0, I)
random_latents = torch.rand(5, latent_dim).to(device)

# Decode the random latent vectors
random_decoded = model(random_latents)
random_decoded = random_decoded.view(-1, 28, 28)

# 2. Select 5 latent vectors from the test set
test_latents = initial_latents[:5]

# Decode the latent vectors from the test set
test_decoded = model(test_latents)
test_decoded = test_decoded.view(-1, 28, 28)

# Save the images
plot_images(random_decoded, "random_latents_images", "Images from Random Latents (U(0, I))")
plot_images(test_decoded, "test_set_latents_images", "Images from Test Set Latents")
import torch
from AutoDecoder import AutoDecoder
from utils import create_dataloaders, plot_tsne
from evaluate import evaluate_model

# Load the dataset
train_ds, train_dl, test_ds, test_dl = create_dataloaders(data_path='dataset', batch_size=64)

# Debugging: Check if the datasets are loaded properly
print(f"Training dataset size: {len(train_ds)}")
print(f"Test dataset size: {len(test_ds)}")
print(f"Number of batches in training DataLoader: {len(train_dl)}")
print(f"Number of batches in test DataLoader: {len(test_dl)}")

# If data is loaded correctly, continue with the training process...
if len(train_dl) == 0:
    print("Training DataLoader is empty!")
    exit()

# Initialize the AutoDecoder model
latent_dim = 64
image_size = 28 * 28
model = AutoDecoder(latent_dim=latent_dim, image_size=image_size)

# Set up the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training Loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for idx, (i, data) in enumerate(train_dl):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        reconstructions = model(data)
        loss = criterion(reconstructions, data)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_dl):.4f}")

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    initial_latents = torch.randn(len(test_ds), latent_dim, device=device)
    optimizer_test = torch.optim.Adam([initial_latents.requires_grad_()], lr=0.001)
    test_loss = evaluate_model(model, test_dl, optimizer_test, initial_latents, epochs=10, device=device)
    print(f"Test set evaluation loss: {test_loss:.4f}")

# Plot t-SNE visualization of the latent space
plot_tsne(test_ds, initial_latents, file_name='tsne_plot.png')

import torch
from AutoDecoder import AutoDecoder
from utils import create_dataloaders, plot_tsne
from evaluate import evaluate_model

# Load the dataset
train_ds, train_dl, test_ds, test_dl = create_dataloaders(data_path='dataset', batch_size=64)

# Check if the dataset was loaded correctly
print(f"Training dataset size: {len(train_ds)}")  # Ensure train_ds is not empty
print(f"Number of batches in training DataLoader: {len(train_dl)}")  # Ensure DataLoader has batches

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

    # Ensure you don't divide by zero if len(train_dl) is zero
    if len(train_dl) > 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_dl):.4f}")
    else:
        print("Warning: No batches in DataLoader")

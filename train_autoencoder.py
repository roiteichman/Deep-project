import torch
from AutoDecoder import AutoDecoder
from utils import create_dataloaders, plot_tsne
from evaluate import evaluate_model
import pandas as pd


# Define a preprocessing function for the dataset
def preprocess_dataset(file_path):
    data = pd.read_csv(file_path)

    # Convert the data to numeric values (this is an extra precaution)
    data = data.apply(pd.to_numeric, errors='coerce')

    # Ensure it's in float32 format to match PyTorch expectations
    data = data.fillna(0).astype(float)

    return data


# Load the dataset
train_data_path = 'dataset/fashion-mnist_train.csv'
test_data_path = 'dataset/fashion-mnist_test.csv'

# Preprocess the datasets
preprocess_dataset(train_data_path)
preprocess_dataset(test_data_path)

# Load the data using the existing utils.py function
train_ds, train_dl, test_ds, test_dl = create_dataloaders(data_path='dataset', batch_size=64)

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

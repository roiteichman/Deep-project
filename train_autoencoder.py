import torch
from AutoDecoder import AutoDecoder
from utils import create_dataloaders, plot_tsne
from evaluate import evaluate_model
import pandas as pd
import os


# Preprocess the CSV file to ensure it has the correct numeric data types
def preprocess_csv(file_path):
    data = pd.read_csv(file_path)

    # Convert all columns to numeric (use coerce to handle any potential non-numeric values)
    data = data.apply(pd.to_numeric, errors='coerce')

    # Fill any NaN values with 0 (or an appropriate value)
    data = data.fillna(0)

    # Ensure that all data is in float32 format
    return data.astype(float)


# Save the preprocessed CSV back
def save_preprocessed_csv(data, file_path):
    data.to_csv(file_path, index=False)


# Preprocess both the training and test CSV files
train_data_path = 'dataset/fashion-mnist_train.csv'
test_data_path = 'dataset/fashion-mnist_test.csv'

# Preprocess the CSV files
train_data = preprocess_csv(train_data_path)
test_data = preprocess_csv(test_data_path)

# Save the preprocessed files (optional: overwrite the originals or save as new files)
preprocessed_train_data_path = 'dataset/preprocessed_fashion-mnist_train.csv'
preprocessed_test_data_path = 'dataset/preprocessed_fashion-mnist_test.csv'
save_preprocessed_csv(train_data, preprocessed_train_data_path)
save_preprocessed_csv(test_data, preprocessed_test_data_path)

# Now use the preprocessed CSV files in the create_dataloaders function
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

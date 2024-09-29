import torch
import torch.nn as nn

class AutoDecoder(nn.Module):
    def __init__(self, latent_dim=64, image_size=28*28):
        """
        Initialize the AutoDecoder with the latent space dimension and output image size.
        :param latent_dim: Dimensionality of the latent space.
        :param image_size: Flattened size of the image (28x28 for Fashion MNIST).
        """
        super(AutoDecoder, self).__init__()

        # Store the latent dimension and image size
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Define the decoder network, which will map the latent space to the image space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),  # First layer
            nn.ReLU(),
            nn.Linear(128, 256),  # Second layer
            nn.ReLU(),
            nn.Linear(256, 512),  # Third layer
            nn.ReLU(),
            nn.Linear(512, image_size),  # Final output layer
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, z):
        """
        Forward pass of the AutoDecoder. Takes in the latent vector and decodes it to an image.
        :param z: Latent vector of size (batch_size, latent_dim)
        :return: Decoded image of size (batch_size, image_size) in the range [0, 255]
        """
        return self.decoder(z) * 255  # Scale the output back to [0, 255]

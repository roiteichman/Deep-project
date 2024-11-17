import torch
import torch.nn as nn

class VAD_Decoder(torch.nn.Module):
    def __init__(self, latent_dim=128):
        super(VAD_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),  # Map to 256 channels with 7x7 spatial dimensions
            nn.ReLU(inplace=True),
            
            # Reshape output from the linear layer to start the convolutional decoding
            nn.Unflatten(1, (256, 7, 7)),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.decoder(x)
        return out * 255.0
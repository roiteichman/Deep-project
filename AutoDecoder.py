import torch
import torch.nn as nn


class AutoDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, latents):
        return self.decoder(latents.view(latents.size(0), self.latent_dim, 1,1)).squeeze(1)
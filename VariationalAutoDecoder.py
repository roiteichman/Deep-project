import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import create_dataloaders

class VariationalAutoDecoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(VariationalAutoDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        
        # Prior parameters (learnable)
        self.prior_mu = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(latent_dim))

        layers = []
        in_dim = latent_dim
        for h_dim in self.hidden_dim:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim

        # Final output layer to reconstruct data
        layers.append(nn.Linear(in_dim, self.output_dim))
        layers.append(nn.Sigmoid())  # Normalized output between 0 and 1 for image reconstruction

        self.decoder = nn.Sequential(*layers)
        
        # Inference network (q(z|x)) parameters
        self.inference_mu = nn.Parameter(torch.zeros(latent_dim, device=self.device))
        self.inference_logvar = nn.Parameter(torch.zeros(latent_dim, device=self.device))

    def decode(self, z):
        return self.decoder(z)*255.0
    
    def sample_latent(self, batch_size, sample_inference=False):
        if sample_inference:
            mu = self.inference_mu
            logvar = self.inference_logvar
        else:
            mu = self.prior_mu
            logvar = self.prior_logvar
            
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(batch_size, self.latent_dim).to(self.device)
        return mu + eps * std
    
    def generate(self, batch_size=1):
        with torch.no_grad():
            z = self.sample_latent(batch_size, sample_inference=False)
            return self.decode(z)
        
    def forward(self, batch_size=64):
        # Sample from inference distribution during training
        z = self.sample_latent(batch_size, sample_inference=True)
        return self.decode(z)
    
    def kl_divergence(self):
        """Compute KL divergence between inference and prior distributions"""
        return 0.5 * torch.sum(
            torch.exp(self.inference_logvar - self.prior_logvar) +
            (self.inference_mu - self.prior_mu).pow(2) / torch.exp(self.prior_logvar) -
            1 + self.prior_logvar - self.inference_logvar
        )
    

def vad_loss(model, recon_x, x, beta=1.0):
    """
    Compute VAD loss: reconstruction loss + KL divergence
    beta: weight of the KL divergence term (for beta-VAE like behavior)
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence between inference and prior
    kl_div = model.kl_divergence()
    
    return recon_loss + beta * kl_div

def train_vad(model, train_loader, num_epochs=100, learning_rate=1e-3, beta=1.0):
    print('Starting training...')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        total_loss = 0
        for batch_idx, (i, data) in enumerate(train_loader):
            batch_size = i.size(0)
            data = data.view(batch_size, -1).to(model.device)
            
            optimizer.zero_grad()
            
            # Generate reconstruction
            recon_batch = model(batch_size)
            
            # Compute loss
            loss = vad_loss(model, recon_batch, data.float(), beta)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}: Average loss = {avg_loss:.4f}')

    return avg_loss

def start_training():
    model = VariationalAutoDecoder(latent_dim=128, hidden_dim=[400], output_dim=784)
    data_path='dataset'
    batch_size=64
    train_ds, train_dl, test_ds, test_dl = create_dataloaders(data_path=data_path, batch_size=batch_size)
    train_vad(model, train_dl, num_epochs=10, learning_rate=1e-3, beta=1.0)
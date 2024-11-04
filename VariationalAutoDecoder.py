import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import create_dataloaders
from evaluate import reconstruction_loss, evaluate_model
from utils import plot_tsne

class VariationalAutoDecoder(nn.Module):
    def __init__(self, latent_dim=128, data_path='dataset', batch_size=64, lr=0.005):
        super(VariationalAutoDecoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.train_ds, self.train_dl, self.test_ds, self.test_dl = create_dataloaders(data_path=data_path, batch_size=self.batch_size)
        self.num_samples_in_dataset = len(self.train_ds)
        self.learning_rate = lr
        
        self.mus = torch.randn(self.num_samples_in_dataset, self.latent_dim, requires_grad=True, device=self.device)
        self.log_vars = torch.randn(self.num_samples_in_dataset, self.latent_dim, requires_grad=True, device=self.device)

        self.decoder = Decoder(latent_dim=self.latent_dim).to(self.device)

        self.optimizer = torch.optim.Adam([{'params': self.parameters()}, {'params': self.mus}, {'params': self.log_vars}], lr=self.learning_rate)
        self.to(self.device)

        
    def _reparameterize(self, mu, log_var):
        """instead of directly learning the variance, 
        it’s common practice to learn log_var (logarithm of the variance).
        This avoids numerical instability because directly 
        optimizing the variance can lead to extremely small or large values"""

        # sigma = sqrt(var)
        # sigma = sqrt(exp(log_var)) = (exp(log_var))^0.5= exp(0.5 * log_var)
        sigma = torch.exp(0.5 * log_var).to(self.device) 
        epsilon = torch.randn_like(sigma).to(self.device)  # Sample from standard normal with same shape as sigma
        #z=μ+σ⋅ϵ

        return mu + sigma * epsilon
        

    def forward(self, z):
        # Decode to generate output
        # z = z.view(z.size(0), self.latent_dim, 1, 1).to(self.device)
        output = self.decoder(z)  
        output = torch.squeeze(output, 1)  # Remove the extra channel dimension to match the target size (batch_size, 28, 28)
        return output * 255.0 # Scale output to [0, 255]

    def vad_loss(self, recon_x, x, mu: torch.Tensor, log_var: torch.Tensor, beta: float=3.8):
        # Reconstruction loss
        recon_loss = reconstruction_loss(x=x, x_rec=recon_x)
        
        # KL Divergence
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # sigma = sqrt(var) => sigma^2 = exp(log(sigma^2)) = exp(log_var)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + beta * kl_divergence

    # Training the VAD model
    def train_model(self, num_epochs=100, beta=3.8):
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (i, data) in enumerate(self.train_dl):
                data = data.float().to(self.device)

                self.optimizer.zero_grad()
                
                mu = self.mus[i].to(self.device)
                log_var = self.log_vars[i].to(self.device)
                z = self._reparameterize(mu, log_var).to(self.device)
                # Forward pass
                recon_x = self(z)
                
                # Compute loss
                loss = self.vad_loss(recon_x=recon_x, x=data, mu=mu, log_var=log_var, beta=beta)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dl)        
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(self.train_dl):.4f}')

        return avg_loss

    def test_vad(self, num_epochs=100, learnig_rate=0.005):
        self.eval()
        self.test_latents = torch.randn(len(self.test_ds), self.latent_dim, requires_grad=True, device=self.device)
        optimizer_test = torch.optim.Adam([{'params': self.test_latents}], lr=learnig_rate)
        test_loss = evaluate_model(self, self.test_dl, optimizer_test, self.test_latents, num_epochs, self.device)
        return test_loss
    
    def plot_tsne(self):
        print("Generating t-SNE plot...")
        plot_tsne(self.test_ds, self.test_latents, file_name='tsne_plot_VAD.png', plot_title="t-SNE Visualization of Latents")

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
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
        return out

# model = VariationalAutoDecoder()
# train_loss = model.train_model(num_epochs=100)
# print(f'Training loss: {train_loss:.4f}')
# test_loss = model.test_vad(num_epochs=100)
# print(f'Test loss: {test_loss:.4f}')

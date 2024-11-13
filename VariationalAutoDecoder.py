import torch
import torch.nn as nn
from utils import create_dataloaders
from evaluate import evaluate_model
from utils import plot_tsne
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from Decoder import Decoder


class VariationalAutoDecoder(nn.Module, ABC):
    def __init__(self, latent_dim=128, data_path='dataset', batch_size=64, lr=0.005):
        super(VariationalAutoDecoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.train_ds, self.train_dl, self.test_ds, self.test_dl = create_dataloaders(data_path=data_path, 
                                                                                      batch_size=self.batch_size)
        self.num_samples_in_dataset = len(self.train_ds)
        self.learning_rate = lr

        self.decoder = Decoder(latent_dim=self.latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=self.learning_rate)

        self.to(self.device)

    @abstractmethod
    def _reparameterize(self, mu, log_var):
        pass
        

    def forward(self, z):
        # Decode to generate output
        output = self.decoder(z)  
        output = torch.squeeze(output, 1)  # Remove the extra channel dimension to match the target size (batch_size, 28, 28)
        return output * 255.0 # Scale output to [0, 255]
    
    @abstractmethod
    def _get_kl_divergence(self, indices):
        pass
    
    def _vad_loss(self, recon_x, x, indices, beta: float=3.8):
        # Reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
        
        # KL Divergence
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # sigma = sqrt(var) => sigma^2 = exp(log(sigma^2)) = exp(log_var)
        kl_divergence = self._get_kl_divergence(indices)

        total_loss = recon_loss + beta * kl_divergence
        
        return total_loss, recon_loss, kl_divergence

    @abstractmethod
    def _get_latent_vectors(self, indices):
        pass


    # Training the VAD model
    def train_model(self, num_epochs=100, beta=3.8, lr=0.005, verbose=True):
        print("Training the VAD model...")
        self.train()
        total_losses = []
        recon_losses = []
        kl_losses = []
        self.learning_rate = lr
        for epoch in range(num_epochs):
            total_loss = 0
            total_recon_loss = 0
            total_kl_loss = 0
            for batch_idx, (i, data) in enumerate(self.train_dl):
                data = data.float().to(self.device)

                self.optimizer.zero_grad()
                
                z = self._get_latent_vectors(indices=i)
                # Forward pass
                recon_x = self(z)
                
                # Compute loss
                loss, recon_loss, kl_loss = self._vad_loss(recon_x=recon_x, x=data, indices=i, beta=beta)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()

            avg_total_loss = total_loss / len(self.train_dl)    
            avg_recon_loss = total_recon_loss / len(self.train_dl)
            avg_kl_loss = total_kl_loss / len(self.train_dl)

            total_losses.append(avg_total_loss)
            recon_losses.append(avg_recon_loss)
            kl_losses.append(avg_kl_loss)   

            if verbose:
                print(f'Epoch {epoch + 1}, Loss: {total_loss / len(self.train_dl):.4f}')
        
        if verbose:
            self.plot_losses(recon_losses, kl_losses, total_losses)

        return avg_total_loss, avg_recon_loss, avg_kl_loss

    def test_vad(self, num_epochs=100, learning_rate=0.005):
        print("Testing the VAD model...")
        self.eval()
        self.test_latents = torch.randn(len(self.test_ds), self.latent_dim, requires_grad=True, 
                                        device=self.device)
        optimizer_test = torch.optim.Adam([{'params': self.test_latents}], lr=learning_rate)
        test_loss = evaluate_model(self, self.test_dl, optimizer_test, self.test_latents, 
                                   num_epochs, self.device)
        return test_loss
    
    def plot_tsne(self):
        print("Generating t-SNE plot...")
        plot_tsne(self.test_ds, self.test_latents, file_name='tsne_plot_VAD.png', plot_title="t-SNE Test Latents - VAD")

    def plot_losses(self, recon_losses, kl_losses, total_losses):

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        # Plot the Reconstruction Loss
        ax1.plot(recon_losses, label='Reconstruction Loss', color='tab:red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Reconstruction Loss')
        ax1.set_title('Reconstruction Loss Over Epochs')
        ax1.set_yscale('log')  # Log scale for better visualization of large differences
        ax1.legend()

        # Plot the KL Divergence Loss
        ax2.plot(kl_losses, label='KL Divergence', color='tab:blue')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('KL Divergence')
        ax2.set_title('KL Divergence Over Epochs')
        ax2.legend()

        # Plot the Total Loss
        ax3.plot(total_losses, label='Total Loss', color='tab:green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Total Loss')
        ax3.set_title('Total Loss Over Epochs')
        ax3.set_yscale('log')  # Log scale to see convergence
        ax3.legend()

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()


class VariationalAutoDecoderNormal(VariationalAutoDecoder):
    def __init__(self, latent_dim=128, data_path='dataset', batch_size=64, lr=0.005):
        super().__init__(latent_dim, data_path, batch_size, lr)
                
        self.mus = torch.randn(self.num_samples_in_dataset, self.latent_dim, requires_grad=True, device=self.device)
        self.log_vars = torch.randn(self.num_samples_in_dataset, self.latent_dim, requires_grad=True, device=self.device)

        self.optimizer = torch.optim.Adam([{'params': self.parameters()}, {'params': self.mus}, {'params': self.log_vars}], lr=self.learning_rate)

    def _get_latent_vectors(self, indices):
        mu = self.mus[indices]
        log_var = self.log_vars[indices]
        return self._reparameterize(mu, log_var)
    
    def _get_kl_divergence(self, indices):
        mu = self.mus[indices]
        log_var = self.log_vars[indices]
        kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_divergence
    
    def _reparameterize(self, mu, log_var):
        """instead of directly learning the variance, 
        it’s common practice to learn log_var (logarithm of the variance).
        This avoids numerical instability because directly 
        optimizing the variance can lead to extremely small or large values"""

        # sigma = sqrt(var)
        # sigma = sqrt(exp(log_var)) = (exp(log_var))^0.5= exp(0.5 * log_var)
        sigma = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)  # Sample from standard normal with same shape as sigma

        return mu + sigma * epsilon
    

    
class VariationalAutoDecoderLaplase(VariationalAutoDecoder):
    def __init__(self, latent_dim=128, data_path='dataset', batch_size=64, lr=0.005):
        super().__init__(latent_dim, data_path, batch_size, lr)
        self.laplace_dist = torch.distributions.Laplace(0, 1)

        self.mus = nn.Parameter(torch.zeros(self.num_samples_in_dataset, self.latent_dim).to(self.device))
        self.log_bs = nn.Parameter(self.laplace_dist.sample((self.num_samples_in_dataset, self.latent_dim)).to(self.device))

        assert self.mus.requires_grad == True
        assert self.log_bs.requires_grad == True
    
        self.optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=self.learning_rate)

    def _get_latent_vectors(self, indices):
        mu = self.mus[indices]
        log_b = self.log_bs[indices]

        return self._reparameterize(mu, log_b)

    def _reparameterize(self, mu, log_b):
        b = torch.exp(log_b)
        epsilon = self.laplace_dist.sample(b.shape).to(self.device)

        return mu + b * epsilon
    
    def _get_kl_divergence(self, indices):
        b = torch.exp(self.log_bs[indices])
        mu = self.mus[indices]
        kl_loss = torch.log(b) + (1 / b) + torch.abs(mu) - 0.5 
        
        # Mean over batch
        return torch.mean(kl_loss)

class VariationalAutoDecoderExponential(VariationalAutoDecoder):
    def __init__(self, latent_dim=128, data_path='dataset', batch_size=64, lr=0.005, lambda_rate=1):
        super().__init__(latent_dim, data_path, batch_size, lr)
        self.lambda_rate = lambda_rate

        self.log_lambda = nn.Parameter(torch.ones(self.num_samples_in_dataset, self.latent_dim).to(self.device))

        assert self.log_lambda.requires_grad == True

    def _get_latent_vectors(self, indices):
        log_lambda = self.log_lambda[indices]
        return self._reparameterize(log_lambda)
    
    def _reparameterize(self, log_lambda):
        lambda_param = torch.exp(log_lambda)
        epsilon = torch.rand_like(lambda_param)
        
        # Reparameterize to get samples from an exponential distribution
        z = -torch.log(1 - epsilon) / lambda_param
        return z
    
    def _get_kl_divergence(self, indices):
        log_lambda = self.log_lambda[indices]
        lambda_param = torch.exp(log_lambda)
        kl_loss = torch.log(lambda_param) - 1

        # Mean over batch
        return torch.mean(kl_loss)
    


model = VariationalAutoDecoderNormal()
train_loss,_,_ = model.train_model(num_epochs=5)
# print(f'Training loss: {train_loss:.4f}')
# test_loss = model.test_vad(num_epochs=100)
# print(f'Test loss: {test_loss:.4f}')

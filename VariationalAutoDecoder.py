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

        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 28*28),
                                     nn.Sigmoid()).to(self.device)
        
        self.optimizer = torch.optim.Adam([{'params': self.parameters()}, {'params': self.mus}, {'params': self.log_vars}], lr=self.learning_rate)

        
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
        

    def forward(self, samples_indices=None, latent_vector=None):
        mu , log_var = None, None
        if samples_indices is not None:
            # mu and log_var tensors for the specific samples indices
            mu = self.mus[samples_indices].to(self.device)
            log_var = self.log_vars[samples_indices].to(self.device)
            
            # Sample latent vector z
            z = self._reparameterize(mu, log_var).to(self.device)
        else:
            z = latent_vector.to(self.device)
        
        # Decode to generate output
        # z = z.view(z.size(0), self.latent_dim, 1, 1).to(self.device)
        output = self.decoder(z) * 255.0  # Scale output to [0, 255]
        # output = output.view(output.size(0), -1)  # Remove the extra channel dimension to match the target size (batch_size, 28, 28)
        return output, mu, log_var

    def vad_loss(self, recon_x, x, mu: torch.Tensor, log_var: torch.Tensor, beta: float=1.0):
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
    def train_model(self, num_epochs=100):
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (i, data) in enumerate(self.train_dl):
                data = data.view(i.size(0), -1)  # Flatten images to vector shape (batch_size, 784)
                data = data.float().to(self.device)

                self.optimizer.zero_grad()
                
                # Forward pass
                recon_x, mu, log_var = self(samples_indices=i)
                
                # Compute loss
                loss = self.vad_loss(recon_x, data, mu, log_var)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dl)        
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(self.train_dl):.4f}')

        return avg_loss

    def test_vad(self, num_epochs=100):
        self.eval()
        self.test_latents = torch.randn(len(self.test_ds), self.latent_dim, requires_grad=True, device=self.device)
        optimizer_test = torch.optim.Adam([{'params': self.test_latents}], lr=self.learning_rate)
        
        test_loss = self._evaluate_model(optimizer_test, num_epochs)
        return test_loss

    def _evaluate_model(self, opt, epochs):
        """
        :param model: the trained model
        :param test_dl: a DataLoader of the test set
        :param opt: a torch.optim object that optimizes ONLY the test set latents
        :param latents: initial values for the latents of the test set
        :param epochs: how many epochs to train the test set latents for
        :return:
        """
        self.to(self.device)
        for epoch in range(epochs):
            for i, x in self.test_dl:
                i = i.to(self.device)
                x = x.float().to(self.device)
                x = x.view(i.size(0), -1)
                x_rec, _, _ = self(latent_vector=self.test_latents[i].to(self.device))
                loss = reconstruction_loss(x, x_rec)
                opt.zero_grad()
                loss.backward()
                opt.step()

        losses = []
        with torch.no_grad():
            for i, x in self.test_dl:
                i = i.to(self.device)
                x = x.float().to(self.device)
                x = x.view(i.size(0), -1)
                x_rec, _, _ = self(latent_vector=self.test_latents[i].to(self.device))
                loss = reconstruction_loss(x, x_rec)
                losses.append(loss.item())

            final_loss = sum(losses) / len(losses)

        return final_loss
    
    def plot_tsne(self):
        print("Generating t-SNE plot...")
        plot_tsne(self.test_ds, self.test_latents, file_name='tsne_plot_VAD.png', plot_title="t-SNE Visualization of Latents")

            

# model = VariationalAutoDecoder()
# train_loss = model.train_model(num_epochs=100)
# print(f'Training loss: {train_loss:.4f}')
# test_loss = model.test_vad(num_epochs=100)
# print(f'Test loss: {test_loss:.4f}')

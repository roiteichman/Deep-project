import torch
import torch.nn as nn
from utils import create_dataloaders, plot_tsne
from evaluate import evaluate_model, reconstruction_loss
import matplotlib.pyplot as plt
import os

class AutoDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels=1, learning_rate=0.005, criterion='reconstruction_loss'):
        super(AutoDecoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.criterion = reconstruction_loss

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, latents):
        out = self.decoder(latents.view(latents.size(0), self.latent_dim, 1,1).to(self.device))
        # Remove the extra channel dimension to match the target size (batch_size, 28, 28)
        out = torch.squeeze(out, 1)

        out = out*255.0
        return out
    
    def train_model(self, train_ds, train_dl, num_epochs=100):
        self.train()
        latent_vectors = torch.randn(len(train_ds), self.latent_dim, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([{'params': self.parameters()}, {'params': latent_vectors}], lr=self.learning_rate)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            total_loss = 0.0

            for idx, (i, data) in enumerate(train_dl):
                latent_batch = latent_vectors[i].to(self.device)
                data = data.float().to(self.device)

                optimizer.zero_grad()
                y_reconstructions = self(latent_batch.to(self.device))
                loss = self.criterion(data, y_reconstructions)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dl)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        
        return avg_loss
    
    def test_model(self, test_ds, test_dl, num_epochs=100):
        self.eval()
        self.test_latents = torch.randn(len(test_ds), self.latent_dim, requires_grad=True, device=self.device)
        optimizer_test = torch.optim.Adam([{'params': self.test_latents}], lr=self.learning_rate)

        test_loss = evaluate_model(self, test_dl, optimizer_test, self.test_latents, num_epochs, self.device)
        
        return test_loss
    
class Trainer:
    def __init__(self, latent_dim=128, batch_size=64, num_epochs=100, data_path='dataset'):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.data_path = data_path

        self.train_ds, self.train_dl, self.test_ds, self.test_dl = create_dataloaders(data_path=self.data_path, batch_size=self.batch_size)
        
        self.model = AutoDecoder(latent_dim=self.latent_dim)
        self.model.to(self.model.device)


    def train_and_evaluate(self):
        train_loss = self.model.train_model(self.train_ds, self.train_dl, self.num_epochs)
        print(f'Training Loss: {train_loss:.4f}')
        
        test_loss = self.model.test_model(self.test_ds, self.test_dl, self.num_epochs)
        print(f'Test Loss: {test_loss:.4f}')
        print('Training and Evaluation Complete!')

    def plot_images(self, images, file_name, title):
        print("Saving images...")
        os.makedirs('output_images', exist_ok=True)
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i, ax in enumerate(axes):
            ax.imshow(images[i].cpu().detach().numpy(), cmap='gray')
            ax.axis('off')
        plt.suptitle(title)
        output_path = f"output_images/{file_name}.png"
        plt.savefig(output_path)
        print(f"Images saved to {output_path}")
        plt.close()

    def test_samples(self):
        print("Generating random samples...")
        random_latents = torch.rand(5, self.latent_dim).to(self.model.device)
        random_decoded = self.model(random_latents).view(-1, 28, 28)

        test_latents = self.model.test_latents[:5]
        test_decoded = self.model(test_latents).view(-1, 28, 28)

        self.plot_images(random_decoded, "random_latents_images", "Images from Random Latents (U(0, I))")
        self.plot_images(test_decoded, "test_set_latents_images", "Images from Test Set Latents")

    def plot_tsne(self):
        print("Generating t-SNE plot...")
        plot_tsne(self.test_ds, self.model.test_latents, file_name='tsne_plot.png', plot_title="t-SNE Visualization of Latents")

# trainer = Trainer()
# trainer.train_and_evaluate()
# trainer.test_samples()
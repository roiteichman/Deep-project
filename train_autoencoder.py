import torch
from AutoDecoder import AutoDecoder
from utils import create_dataloaders, plot_tsne
from evaluate import evaluate_model, reconstruction_loss
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
class Trainer:
    def __init__(self, latent_dim=128, batch_size=64, num_epochs=20, learning_rate=0.0005 ,weight_decay=1e-4, latent_learning_rate=0.0005):
        self.latent_dim = latent_dim  # Increase latent dimension to 128
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.latent_learning_rate = latent_learning_rate
        self.weight_decay = weight_decay

        self.train_ds, self.train_dl, self.test_ds, self.test_dl = create_dataloaders(data_path='dataset', batch_size=batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoDecoder(latent_dim=latent_dim).to(self.device)
        self.criterion = torch.nn.MSELoss()
        #self.criterion = reconstruction_loss



    def train_autodecoder(self, seperate_optimizers_and_schedulers=True):
        self.model.train()
        latent_vectors = torch.randn(len(self.train_ds), self.latent_dim, requires_grad=True, device=self.device)
        optimizers, schedulers = [], []

        if seperate_optimizers_and_schedulers:
            net_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            latent_optimizer = torch.optim.Adam([latent_vectors], lr=self.latent_learning_rate)
            net_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(net_optimizer, mode='min',
                factor=0.5, patience=10, verbose=True)
            latent_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(latent_optimizer, mode='min',
                factor=0.5, patience=10, verbose=True)
            optimizers = [net_optimizer, latent_optimizer]
            schedulers = [net_scheduler, latent_scheduler]
        else:
            optimizer = torch.optim.Adam([{'params': self.model.parameters()}, {'params': latent_vectors}], lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            optimizers = [optimizer]
            schedulers = [scheduler]


        # Training Loop
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}:')
            total_loss = 0.0

            for idx, (i, data) in enumerate(self.train_dl):
                # Get the corresponding latent vectors for the batch
                latent_batch = latent_vectors[i].to(self.device)

                # very important to normalize the data
                data = data.float() / 255.0

                data = data.to(self.device)

                for optimizer in optimizers:
                    optimizer.zero_grad()
                y_reconstructions = self.model(latent_batch)
                loss = self.criterion(data.float(), y_reconstructions)
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dl)
            for scheduler in schedulers:
                scheduler.step(total_loss)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
        
        return avg_loss

    def test_autodecoder(self):
        self.model.eval()
        total_loss = 0.0
        # progress_bar = tqdm(self.test_dl, desc="Testing")

        initial_latents = torch.randn(len(self.test_ds), self.latent_dim, device=self.device)
        optimizer_test = torch.optim.Adam([initial_latents], lr=self.learning_rate)
        loss = evaluate_model(self.model, self.test_dl, optimizer_test, initial_latents, self.num_epochs, self.device)

        return loss

    def train_and_evaluate(self, seperate_optimizers_and_schedulers=True):
        train_loss = self.train_autodecoder(seperate_optimizers_and_schedulers)
        print(f'Training Loss: {train_loss:.4f}')
        test_loss = self.test_autodecoder()
        print(f'Test Loss: {test_loss:.4f}')




def plot_images(images, file_name, title):
    # Create the output directory if it doesn't exist
    os.makedirs('output_images', exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i].cpu().detach().numpy(), cmap='gray')
        ax.axis('off')

    plt.suptitle(title)

    # Save the plot as a PNG file
    output_path = f"output_images/{file_name}.png"
    plt.savefig(output_path)
    print(f"Images saved to {output_path}")
    plt.close()

def test_samples():
    # 1. Sample 5 random latent vectors from U(0, I)
    random_latents = torch.rand(5, latent_dim).to(device)

    # Decode the random latent vectors
    random_decoded = model(random_latents)
    random_decoded = random_decoded.view(-1, 28, 28)

    # 2. Select 5 latent vectors from the test set
    test_latents = initial_latents[:5]

    # Decode the latent vectors from the test set
    test_decoded = model(test_latents)
    test_decoded = test_decoded.view(-1, 28, 28)

    # Save the images
    plot_images(random_decoded, "random_latents_images", "Images from Random Latents (U(0, I))")
    plot_images(test_decoded, "test_set_latents_images", "Images from Test Set Latents")

Trainer().train_and_evaluate()
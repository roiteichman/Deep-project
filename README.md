# Auto-Decoders & Variational Auto-Decoders (AD/VAD)

A PyTorch implementation of **Auto-Decoders (AD)** and **Variational Auto-Decoders (VAD)** for generative modeling. Unlike standard Auto-Encoders, this project explores the "Decoder-Only" paradigm, where latent vectors are optimized directly for each training example during backpropagation, eliminating the need for an inference network (Encoder).

## 🧠 Project Overview

This repository investigates two generative architectures applied to the **FashionMNIST** dataset:

1.  **Auto-Decoder (AD):** Deterministic mapping from a learnable latent space to image space.
2.  **Variational Auto-Decoder (VAD):** A probabilistic approach where each sample is associated with a distribution in latent space, regularized by the Kullback-Leibler (KL) divergence (similar to VAEs but without an encoder).

Key concepts explored:
* **Latent Space Optimization:** Directly learning the latent representation $z$ for each datum $x$.
* **Generative Sampling:** generating novel images from the learned prior distribution.
* **Manifold Interpolation:** Smoothly transitioning between object classes (e.g., Trousers $\to$ Dress) to verify latent space continuity.
* **Dimensionality Reduction:** Visualizing class separation using **t-SNE**.

## 📂 Code Structure

The repository is structured as follows:

* **`AutoDecoder.py`**: Implementation of the deterministic Auto-Decoder. Includes the optimization loop for updating specific latent vectors corresponding to batch indices.
* **`VariationalAutoDecoder.py`**: Implementation of the VAD. Handles the reparameterization trick and the ELBO loss function (Reconstruction + KL Divergence).
* **`VAD_Decoder.py`**: The neural architecture for the decoder module (ConvTranspose2d blocks).
* **`evaluate.py`**: Utility functions for calculating reconstruction loss and handling the test-time optimization (since there is no encoder, test samples require a short optimization phase to find their latent $z$).
* **`utils.py`**: Data loading (FashionMNIST), preprocessing, and visualization tools (t-SNE plots).
* **Reports:**
    * [`wet.pdf`](./wet.pdf) - Full project report with experimental results and visualizations.
    * [`dry.pdf`](./dry.pdf) - Theoretical analysis of vanishing gradients and VAE mathematics.

## 🛠️ Installation & Requirements

1.  Clone the repository:
    ~~~bash
    git clone https://github.com/your-username/auto-decoder-generative-model.git
    cd auto-decoder-generative-model
    ~~~

2.  Install dependencies:
    ~~~bash
    pip install torch torchvision numpy pandas matplotlib scikit-learn
    ~~~

## 🚀 Usage

To train the models and reproduce the results, you can import the classes and run the training loop.

**Example for Variational Auto-Decoder:**

~~~python
import torch
from VariationalAutoDecoder import VariationalAutoDecoder

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoDecoder(latent_dim=128, data_path='./data')

# Train (optimizes both Decoder weights and Latent vectors)
model.fit_and_train(num_epochs=100, beta=2.0)

# Generate new samples
model.get_random_samples_images()
~~~

**Testing on New Data:**

Since there is no encoder, inference on the test set involves freezing the decoder and optimizing $z$ vectors to match the test images:

~~~python
from evaluate import evaluate_model

# Run test-time optimization (finding z for new images)
# optimizer: optimizer for the latents
# latents: initial random latents for test set
test_loss = evaluate_model(model.decoder, model.test_dl, optimizer, latents, epochs=10, device=device)
~~~

## 📊 Results

### Latent Space Visualization (t-SNE)
We visualized the learned latent space using t-SNE. The results demonstrate clear clustering of semantic classes (e.g., footwear vs. clothing), indicating that the Auto-Decoder successfully learned a meaningful manifold structure.

### Interpolation
By interpolating linearly between the latent vectors of two different classes (e.g., a Trouser image and a Dress image), the model generates smooth semantic transitions, confirming the density of the learned latent space.

*(See [`wet.pdf`](./wet.pdf) for detailed plots and figures)*

## 📜 Credits & References
This project was implemented as part of the "Deep Learning on Computational Accelerators" course at the Technion.

**Authors:**
* **Roi Teichman**
* **Elad Sznaj**

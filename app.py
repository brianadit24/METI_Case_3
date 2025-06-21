import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# VAE Model
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # Mean
        self.fc22 = nn.Linear(400, 20)  # Log-variance
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar

# Load model
model = VAE()
model.load_state_dict(torch.load("vae_mnist.pth", map_location="cpu"))
model.eval()

st.title("Handwritten Digit Generator")
digit = st.selectbox("Choose a digit (0-9)", list(range(10)))

if st.button("Generate Images"):
    images = []
    for _ in range(5):
        z = torch.randn(1, 20)
        sample = model.decode(z).detach().view(28, 28)
        images.append(sample)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)

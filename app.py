import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Load model architecture
class VAE(torch.nn.Module):
    # (same as above VAE class)

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

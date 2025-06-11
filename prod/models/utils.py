import io
import os
from math import prod


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder

class PokemonDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Codificar el tipo primario con LabelEncoder
        self.label_encoder = LabelEncoder()
        self.encoded_types = self.label_encoder.fit_transform(self.img_labels.iloc[:, 1])  # Segunda columna: tipo 1

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.encoded_types[idx]
        pokemon_name = os.path.splitext(os.path.basename(img_path))[0].lower()
        return image, label,pokemon_name

class AUG_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1):
        super(AUG_block, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, strides, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, X):
        return self.block(X)

class DEC_block(nn.Module):
    def __init__(self, out_channels, in_channels=3, kernel_size=4, strides=2,
                 padding=1, alpha=0.2):
        super(DEC_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha, inplace=True)
        )

    def forward(self, X):
        return self.block(X)

# VAE con los nuevos bloques
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder con DEC_block
        self.enc1 = DEC_block(32, 3)
        self.enc2 = DEC_block(64, 32)
        self.enc3 = DEC_block(128, 64)
        self.enc4 = DEC_block(256, 128)

        # Flatten y capas lineales para mu y logvar
        self.fc_mu = nn.Linear(256 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(256 * 7 * 7, latent_dim)

        # Decoder lineal y reshape
        self.fc_dec = nn.Linear(latent_dim, 256 * 7 * 7)

        # Decoder con AUG_block
        self.dec1 = AUG_block(128, 256)
        self.dec2 = AUG_block(64, 128)
        self.dec3 = AUG_block(32, 64)
        self.dec4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)  # salida final
        # No usamos BatchNorm ni activación aquí, solo tanh después

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 256, 7, 7)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = torch.tanh(x)  # Normalizado en [-1, 1]
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def load_model():
    # Ruta local al archivo dentro del proyecto
    checkpoint_path = "prod/models/vae_2000_epocs.pt"

    # Cargar el checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # O 'cuda' si tenés GPU

    # Crear el modelo
    latent_dim = checkpoint['config']['latent_dim']
    model = ConvVAE(latent_dim=latent_dim)

    # Cargar pesos
    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def load_dataset():
    
    csv_path = "data/pokemon.csv"
    img_path = "data/pokemon"

    transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # a [-1, 1]
    ])

    dataset = PokemonDataset(csv_path, img_path, transform=transform)

    return dataset

# def generate_by_neighbors_image(model, dataset, device=None, k=5):
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#     model.eval()
#     model.to(device)

#     all_embeddings = []
#     all_images = []

#     # Extraer embeddings
#     with torch.no_grad():
#         for x, _ in dataset:
#             x = x.unsqueeze(0).to(device)
#             mu, logvar = model.encode(x)
#             z = model.reparameterize(mu, logvar)
#             all_embeddings.append(z.squeeze(0).cpu())
#             all_images.append(x.squeeze(0).cpu())

#     all_embeddings = torch.stack(all_embeddings)  # (N, latent_dim)
#     all_images = torch.stack(all_images)          # (N, C, H, W)

#     idx = torch.randint(len(all_embeddings), (1,)).item()
#     ref_z = all_embeddings[idx]

#     dists = F.pairwise_distance(ref_z.unsqueeze(0), all_embeddings)
#     dists[idx] = float('inf')  # excluirse a sí mismo
#     closest_idx = torch.topk(dists, k=k, largest=False).indices
#     mean_z = all_embeddings[closest_idx].mean(dim=0)

#     with torch.no_grad():
#         gen_img = model.decode(mean_z.unsqueeze(0).to(device)).cpu()
#         gen_img = (gen_img + 1) / 2  # escalar a [0,1]

#     # Devolver imagen generada como Tensor
#     return gen_img[0]  # (C, H, W)
def generate_by_neighbors_image(model, dataset, selected_pokemon, device=None, k=5):
    import torch.nn.functional as F

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    all_embeddings = []
    all_images = []
    all_names = []

    # Extraer embeddings
    with torch.no_grad():
        for x, _, name in dataset:  # ignoramos el tipo
            x = x.unsqueeze(0).to(device)
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            all_embeddings.append(z.squeeze(0).cpu())
            all_images.append(x.squeeze(0).cpu())
            all_names.append(name)

    all_embeddings = torch.stack(all_embeddings)
    all_images = torch.stack(all_images)

    try:
        name_to_match = str(selected_pokemon).lower().strip() 
        idx = all_names.index(name_to_match)
    except ValueError:
        raise ValueError(f"No se encontró el Pokémon '{selected_pokemon}' en el dataset.")

    ref_z = all_embeddings[idx]

    dists = F.pairwise_distance(ref_z.unsqueeze(0), all_embeddings)
    dists[idx] = float('inf')

    closest_idx = torch.topk(dists, k=k, largest=False).indices
    mean_z = all_embeddings[closest_idx].mean(dim=0)

    with torch.no_grad():
        gen_img = model.decode(mean_z.unsqueeze(0).to(device)).cpu()
        gen_img = (gen_img + 1) / 2

    return gen_img[0]
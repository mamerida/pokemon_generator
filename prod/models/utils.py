import io
import os
from math import prod


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import random

class PokemonDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            label = self.img_labels.iloc[idx, 1] #self.encoded_types[idx]
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
    checkpoint_path = "prod/models/vae_4000_epocs.pt"

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

def buscar_imagen_por_nombre(nombre_pokemon, carpeta="data/pokemon"):
    """
    Busca y devuelve la ruta a la imagen de un Pokémon según su nombre en minúsculas.
    """
    nombre = nombre_pokemon.lower().strip()
    for archivo in os.listdir(carpeta):
        if archivo.lower().startswith(nombre) and archivo.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(carpeta, archivo)
    return None

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
    dists[idx] = float('inf')  # Evitar que se seleccione a sí mismo

    closest_idx = torch.topk(dists, k=k, largest=False).indices
    # Incluir también el Pokémon seleccionado en la media
    combined_embeddings = torch.cat([ref_z.unsqueeze(0), all_embeddings[closest_idx]], dim=0)
    mean_z = combined_embeddings.mean(dim=0)

    with torch.no_grad():
        gen_img = model.decode(mean_z.unsqueeze(0).to(device)).cpu()
        gen_img = (gen_img + 1) / 2
    
    neighbor_names = [all_names[i] for i in closest_idx]

    return gen_img[0], neighbor_names

def generate_multiple_by_neighbors(model, dataset, selected_pokemon, device=None, k=5, num_outputs=4):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    all_embeddings = []
    all_images = []
    all_names = []

    with torch.no_grad():
        for x, _, name in dataset:
            x = x.unsqueeze(0).to(device)
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            all_embeddings.append(z.squeeze(0).cpu())
            all_images.append(x.squeeze(0).cpu())
            all_names.append(name)

    all_embeddings = torch.stack(all_embeddings)
    all_images = torch.stack(all_images)

    try:
        idx = all_names.index(str(selected_pokemon).lower().strip())
    except ValueError:
        raise ValueError(f"No se encontró el Pokémon '{selected_pokemon}' en el dataset.")

    ref_z = all_embeddings[idx]
    dists = F.pairwise_distance(ref_z.unsqueeze(0), all_embeddings)
    dists[idx] = float('inf')

    closest_idx = torch.topk(dists, k=k, largest=False).indices
    closest_embeddings = all_embeddings[closest_idx]

    # Generar múltiples combinaciones aleatorias
    generated_imgs = []
    with torch.no_grad():
        for _ in range(num_outputs):
            # Tomar una muestra aleatoria de los vecinos
            sampled_idx = random.sample(range(k), k=min(3, k))  # por ejemplo, usar 3 vecinos por combinación
            sampled_z = closest_embeddings[sampled_idx]
            combined_z = torch.cat([ref_z.unsqueeze(0), sampled_z], dim=0)
            mean_z = combined_z.mean(dim=0)

            gen_img = model.decode(mean_z.unsqueeze(0).to(device)).cpu()
            gen_img = (gen_img + 1) / 2
            generated_imgs.append(to_pil_image(gen_img[0]))

    neighbor_names = [all_names[i] for i in closest_idx]
    return generated_imgs, neighbor_names

def interpolar_pokemon_por_nombre(model, dataset, nombre1, nombre2, steps=10, device=None):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)

    # Buscar imágenes en el dataset
    def buscar_imagen(nombre):
        for img, label, name in dataset:
            if name == nombre:
                return img
        raise ValueError(f"No se encontró imagen para {nombre}")

    # Transformar imagen a tensor adecuado
    transform = Compose([
        Resize((120, 120)),  # Asegurate de que coincida con el tamaño usado al entrenar
        ToTensor(),
        lambda x: x * 2 - 1  # Normalizar a [-1, 1] si el modelo lo necesita
    ])

    x1 = buscar_imagen(nombre1).to(device)
    x2 = buscar_imagen(nombre2).to(device)

    with torch.no_grad():
        z1, _ = model.encode(x1.unsqueeze(0))
        z2, _ = model.encode(x2.unsqueeze(0))

        interpolated_imgs = []
        for alpha in torch.linspace(0, 1, steps):
            z = (1 - alpha) * z1 + alpha * z2
            img = model.decode(z).cpu()
            img = (img + 1) / 2  # Desnormalizar a [0, 1]
            interpolated_imgs.append(img.squeeze(0))  # Quitar batch dim

    return interpolated_imgs  # Lista de tensores (C, H, W)

def generate_representative_pokemon(model, dataset, tipo_deseado, device=None, sample_size=3):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)

    embeddings = []

    # Recolectar los embeddings solo de los del tipo deseado
    with torch.no_grad():
        filtrados = []
        for img, tipo, _ in dataset:
            if tipo == tipo_deseado:
                img = img.unsqueeze(0).to(device)  # (1, C, H, W)
                z, _ = model.encode(img)
                filtrados.append(z.cpu())

    if len(filtrados) == 0:
        print(f"No se encontraron Pokémon del tipo '{tipo_deseado}'")
        return None

    # Tomar 3 al azar (o menos si hay pocos)
    seleccionados = random.sample(filtrados, k=min(sample_size, len(filtrados)))
    all_z = torch.cat(seleccionados, dim=0)
    mean_z = all_z.mean(dim=0).to(device)

    # Decodificar
    with torch.no_grad():
        gen_img = model.decode(mean_z.unsqueeze(0)).cpu()
        gen_img = (gen_img + 1) / 2  # escalar a [0, 1]

    return to_pil_image(gen_img[0])

####################################Este es un generador que devuelve un pokemon a partir de un tipo (junta todos)################
# def generate_representative_pokemon(model, dataset, tipo_deseado, device=None):
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     model.eval()
#     model.to(device)

#     embeddings = []

#     with torch.no_grad():
#         for img, tipo, _ in dataset:
#             if tipo == tipo_deseado:
#                 img = img.unsqueeze(0).to(device)  # (1, C, H, W)
#                 z, _ = model.encode(img)
#                 embeddings.append(z.cpu())

#     if not embeddings:
#         print(f"No se encontraron Pokémon del tipo '{tipo_deseado}'")
#         return None

#     all_z = torch.cat(embeddings, dim=0)
#     mean_z = all_z.mean(dim=0).to(device)

#     with torch.no_grad():
#         gen_img = model.decode(mean_z.unsqueeze(0)).cpu()
#         gen_img = (gen_img + 1) / 2  # escalar a [0, 1]

#     return to_pil_image(gen_img[0])
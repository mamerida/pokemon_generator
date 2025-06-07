
import torch
import torch.nn as nn
import torch.nn.functional as F

from AUG_DEC_block import DEC_block, AUG_block

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
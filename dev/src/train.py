import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json

def vae_loss(recon_x, x, mu, logvar, beta=0.1):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

def save_checkpoint(model, optimizer, epoch, loss, config, path):
    os.makedirs(path, exist_ok=True)

    # Guardar pesos del modelo y estado del optimizador
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }, os.path.join(path, f"vae_checkpoint_epoch_{epoch}.pt"))

    print(f"✔ Modelo guardado en {path}/vae_checkpoint_epoch_{epoch}.pt")

def train_vae(model, dataloader, epochs, device='cuda', save_path="/prod/models", beta=0.1):
    os.makedirs(save_path, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    config = {
        'latent_dim': model.latent_dim,
        'beta': beta,
        'lr': 1e-4,
        'input_size': model.input_size if hasattr(model, 'input_size') else None,
    }

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for x, _ in dataloader:
            x = x.to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar, beta)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}/{epochs}, Loss: {total_loss:.2f}")

        if epoch % 500 == 0 or epoch == epochs:
            save_checkpoint(model, optimizer, epoch, total_loss, config, save_path)
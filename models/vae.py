import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.shortcut(x))

class Encoder(nn.Module):
    def __init__(self, in_channels=101, hidden_dims=[128, 256, 512], latent_dim=16):
        super().__init__()
        modules = []
        
        # Initial projection
        modules.append(nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1))
        
        in_c = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(ResNetBlock(in_c, h_dim))
            modules.append(nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=2, padding=1))
            in_c = h_dim
            
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, padding=1)
        self.fc_var = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=3, padding=1)

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, out_channels=101, hidden_dims=[512, 256, 128], latent_dim=16):
        super().__init__()
        modules = []
        
        self.init_proj = nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=3, padding=1)
        
        in_c = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(ResNetBlock(in_c, h_dim))
            modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
            modules.append(nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1))
            in_c = h_dim
            
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            ResNetBlock(hidden_dims[-1], hidden_dims[-1]),
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1)
        )

    def forward(self, z, target_shape):
        x = self.init_proj(z)
        x = self.decoder(x)
        if x.shape[2:] != target_shape:
            x = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)
        x = self.final_layer(x)
        return x

class VAE(nn.Module):
    def __init__(self, in_channels=101, hidden_dims=[64, 128, 256], latent_dim=16):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dims, latent_dim)
        self.decoder = Decoder(in_channels, hidden_dims[::-1], latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        target_shape = x.shape[2:]
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z, target_shape)
        return recon_x, mu, log_var

def vae_loss_function(recon_x, x, mu, log_var, kld_weight=0.00025):
    # Reconstruction loss (MSE)
    recons_loss = F.mse_loss(recon_x, x)
    
    # KL Divergence
    # Normalized by the spatial dimensions equivalent to mean over pixels, sum over batch
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=[1, 2, 3]), dim=0)
    
    loss = recons_loss + kld_weight * kld_loss
    return loss, recons_loss, kld_loss

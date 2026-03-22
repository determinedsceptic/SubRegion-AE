import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class Encoder(nn.Module):
    def __init__(self, in_channels=101, hidden_dims=None, embedding_dim=64):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        modules = [nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)]

        in_c = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(ResNetBlock(in_c, h_dim))
            modules.append(nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=2, padding=1))
            in_c = h_dim

        self.encoder = nn.Sequential(*modules)
        self.out_conv = nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        return self.out_conv(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=101, hidden_dims=None, embedding_dim=64):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.init_proj = nn.Conv2d(embedding_dim, hidden_dims[0], kernel_size=3, padding=1)

        modules = []
        in_c = hidden_dims[0]
        for h_dim in hidden_dims:
            modules.append(ResNetBlock(in_c, h_dim))
            modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
            modules.append(nn.Conv2d(h_dim, h_dim, kernel_size=3, padding=1))
            in_c = h_dim
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            ResNetBlock(hidden_dims[-1], hidden_dims[-1]),
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z, target_shape):
        x = self.init_proj(z)
        x = self.decoder(x)
        if x.shape[2:] != target_shape:
            x = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)
        return self.final_layer(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        b, c, h, w = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(b, h, w, c)

        codebook_loss = F.mse_loss(quantized, z_perm.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z_perm)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized = z_perm + (quantized - z_perm).detach()

        encoding_onehot = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = encoding_onehot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        indices = encoding_indices.view(b, h, w)

        return quantized, {
            'vq_loss': vq_loss,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'perplexity': perplexity,
            'indices': indices,
        }


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', embed)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', embed.clone())

    @staticmethod
    def _distributed_all_reduce(tensor):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def forward(self, z):
        b, c, h, w = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.pow(2).sum(dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_z.dtype)

        quantized = torch.matmul(encodings, self.embedding).view(b, h, w, c)

        if self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(dim=0).detach()
                embed_sum = torch.matmul(encodings.t(), flat_z.detach())
                cluster_size = self._distributed_all_reduce(cluster_size)
                embed_sum = self._distributed_all_reduce(embed_sum)

                self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.eps)
                    / (n + self.num_embeddings * self.eps)
                    * n
                )
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                self.embedding.copy_(embed_normalized)

        commitment_loss = F.mse_loss(quantized.detach(), z_perm)
        vq_loss = self.commitment_cost * commitment_loss

        quantized = z_perm + (quantized - z_perm).detach()

        avg_probs = encodings.float().mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        indices = encoding_indices.view(b, h, w)

        return quantized, {
            'vq_loss': vq_loss,
            'codebook_loss': torch.zeros_like(vq_loss),
            'commitment_loss': commitment_loss,
            'perplexity': perplexity,
            'indices': indices,
        }


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels=101,
        hidden_dims=None,
        embedding_dim=64,
        num_embeddings=1024,
        commitment_cost=0.25,
        quantizer='ema',
        ema_decay=0.99,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]

        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            hidden_dims=hidden_dims[::-1],
            embedding_dim=embedding_dim,
        )

        if quantizer == 'ema':
            self.quantizer = VectorQuantizerEMA(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=ema_decay,
            )
        elif quantizer == 'standard':
            self.quantizer = VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
            )
        else:
            raise ValueError(f'Unknown quantizer type: {quantizer}')

    def encode(self, x):
        z_e = self.encoder(x)
        z_q, q_info = self.quantizer(z_e)
        return z_q, q_info

    def decode(self, z_q, target_shape):
        return self.decoder(z_q, target_shape)

    def forward(self, x):
        target_shape = x.shape[2:]
        z_q, q_info = self.encode(x)
        recon_x = self.decode(z_q, target_shape)
        return recon_x, q_info

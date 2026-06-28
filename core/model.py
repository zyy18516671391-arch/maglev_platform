import torch
import torch.nn as nn
import torch.nn.functional as F


class MSA_CAE(nn.Module):
    """Multi-scale self-attention convolutional autoencoder.

    Input shape: [batch, window, magnets, features]. The encoder learns a
    compact state representation for multi-magnet sensor windows, and the
    decoder reconstructs the original sensor window for representation checks.
    """

    def __init__(self, num_magnets=4, feature_dim=6, window=8, latent_dim=32):
        super().__init__()
        self.num_magnets = num_magnets
        self.feature_dim = feature_dim
        self.window = window
        self.in_channels = num_magnets * feature_dim
        hidden = 32

        self.conv_short = nn.Conv1d(self.in_channels, hidden, kernel_size=3, padding=1)
        self.conv_long = nn.Conv1d(self.in_channels, hidden, kernel_size=5, padding=2)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden * 2, num_heads=4, batch_first=True)
        self.spatial_projection = nn.Linear(feature_dim, hidden * 2)
        self.spatial_attn = nn.MultiheadAttention(embed_dim=hidden * 2, num_heads=4, batch_first=True)
        self.to_latent = nn.Sequential(nn.Linear(hidden * 4, latent_dim), nn.Tanh())
        self.decoder_seed = nn.Linear(latent_dim, hidden * self.window)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, self.in_channels, kernel_size=1),
        )

    def forward(self, seq):
        batch = seq.shape[0]
        flat = seq.reshape(batch, self.window, self.in_channels)
        x = flat.permute(0, 2, 1)

        short_feat = F.relu(self.conv_short(x))
        long_feat = F.relu(self.conv_long(x))
        features = torch.cat([short_feat, long_feat], dim=1).permute(0, 2, 1)

        temporal_attended, temporal_attention = self.temporal_attn(
            features,
            features,
            features,
            need_weights=True,
        )

        magnet_tokens = seq.mean(dim=1)
        magnet_tokens = self.spatial_projection(magnet_tokens)
        spatial_attended, spatial_attention = self.spatial_attn(
            magnet_tokens,
            magnet_tokens,
            magnet_tokens,
            need_weights=True,
        )

        latent_input = torch.cat(
            [temporal_attended.mean(dim=1), spatial_attended.mean(dim=1)],
            dim=1,
        )
        latent = self.to_latent(latent_input)

        decoded = self.decoder_seed(latent).reshape(batch, 32, self.window)
        reconstruction = self.decoder(decoded).permute(0, 2, 1)
        reconstruction = reconstruction.reshape(batch, self.window, self.num_magnets, self.feature_dim)

        return {
            "latent": latent,
            "reconstruction": reconstruction,
            "attention": temporal_attention,
            "temporal_attention": temporal_attention,
            "spatial_attention": spatial_attention,
        }


class MaglevModel(nn.Module):
    """Multi-magnet maglev dynamic-response PINN prototype."""

    def __init__(self, num_magnets=4, feature_dim=6, window=8, latent_dim=32):
        super().__init__()
        self.num_magnets = num_magnets
        self.feature_dim = feature_dim
        self.window = window
        self.perception = MSA_CAE(num_magnets, feature_dim, window, latent_dim)
        self.response_head = nn.Sequential(
            nn.Linear(latent_dim + 1 + num_magnets, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
            nn.Linear(96, num_magnets),
        )

    def forward(self, seq, t, currents, return_aux=False):
        encoded = self.perception(seq)
        current_features = currents.reshape(currents.shape[0], self.num_magnets)
        response_in = torch.cat([encoded["latent"], t, current_features], dim=1)
        gaps = F.softplus(self.response_head(response_in)) + 0.02

        if return_aux:
            return {
                "gap": gaps,
                "reconstruction": encoded["reconstruction"],
                "latent": encoded["latent"],
                "attention": encoded["attention"],
                "temporal_attention": encoded["temporal_attention"],
                "spatial_attention": encoded["spatial_attention"],
            }
        return gaps

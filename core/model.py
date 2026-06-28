import torch
import torch.nn as nn
import torch.nn.functional as F


class MSA_CAE(nn.Module):
    """感知层：多尺度注意力自编码特征提取网络"""

    def __init__(self, in_channels=4, window=5, latent_dim=16):
        super().__init__()
        self.window = window
        # 多尺度卷积特征提取
        self.conv_s = nn.Conv1d(in_channels, 16, kernel_size=2, padding=1)
        self.conv_l = nn.Conv1d(in_channels, 16, kernel_size=4, padding=2)
        # 自注意力机制
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.fc = nn.Linear(32, latent_dim)

    def forward(self, x):
        # x shape: [batch, window_size, in_channels]
        x = x.permute(0, 2, 1)
        f_s = torch.relu(self.conv_s(x))[:, :, :self.window]
        f_l = torch.relu(self.conv_l(x))[:, :, :self.window]

        feat = torch.cat([f_s, f_l], dim=1).permute(0, 2, 1)
        attn_out, _ = self.attn(feat, feat, feat)
        return torch.relu(self.fc(attn_out.mean(dim=1)))


class MaglevModel(nn.Module):
    """决策层：ID-PINN 双引擎网络"""

    def __init__(self, in_channels=4, window=5):
        super(MaglevModel, self).__init__()
        self.perception = MSA_CAE(in_channels, window)
        # 输入: 潜变量(16) + 时间t(1) + I1(1) + I2(1) = 19
        self.fc = nn.Sequential(
            nn.Linear(16 + 1 + 2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # 输出预测间隙 x1, x2
        )

    def forward(self, seq, t, I1, I2):
        z = self.perception(seq)
        inp = torch.cat([z, t, I1, I2], dim=1)
        # softplus 保证间隙恒大于 0，并加上基础间隙 epsilon
        return torch.nn.functional.softplus(self.fc(inp)) + 0.05


# Modern multi-magnet implementation. The definitions below intentionally
# override the legacy prototype above while preserving import compatibility.
class MSA_CAE(nn.Module):
    """多尺度自注意力卷积自编码器。

    输入形状为 [batch, window, magnets, features]，编码器提取多电磁铁
    多传感器序列的低维表征，解码器重构原始传感器窗口。
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
        self.attn = nn.MultiheadAttention(embed_dim=hidden * 2, num_heads=4, batch_first=True)
        self.to_latent = nn.Sequential(nn.Linear(hidden * 2, latent_dim), nn.Tanh())
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

        attended, attention = self.attn(features, features, features, need_weights=True)
        latent = self.to_latent(attended.mean(dim=1))

        decoded = self.decoder_seed(latent).reshape(batch, 32, self.window)
        reconstruction = self.decoder(decoded).permute(0, 2, 1)
        reconstruction = reconstruction.reshape(batch, self.window, self.num_magnets, self.feature_dim)

        return {
            "latent": latent,
            "reconstruction": reconstruction,
            "attention": attention,
        }


class MaglevModel(nn.Module):
    """多电磁铁磁浮动力响应 PINN 原型。"""

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
            }
        return gaps

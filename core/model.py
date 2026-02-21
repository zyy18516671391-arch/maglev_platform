import torch
import torch.nn as nn


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
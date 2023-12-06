import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseSwinBlock(nn.Module):
    def __init__(self, dim, heads, window_size, mlp_ratio=4.0, num_dense_blocks=3):
        super(ResidualDenseSwinBlock, self).__init__()

        # Swin Transformer Components
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

        # Dense Connectivity Components
        self.dense_blocks = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1) for _ in range(num_dense_blocks)
        ])
        self.norm_dense = nn.LayerNorm(dim)

    def forward(self, x):
        # Swin Transformer
        identity = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = identity + attn_output

        # MLP
        identity = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = identity + x

        # Dense Connectivity
        dense_outputs = [x]
        for dense_block in self.dense_blocks:
            dense_output = dense_block(dense_outputs[-1])
            dense_outputs.append(dense_output)

        x = torch.cat(dense_outputs, dim=1)
        x = self.norm_dense(x)

        return x


# Example usage
dim = 256
heads = 8
window_size = 7
mlp_ratio = 4.0
num_dense_blocks = 3

residual_dense_swin_block = ResidualDenseSwinBlock(dim, heads, window_size, mlp_ratio, num_dense_blocks)

# Input tensor (batch_size, sequence_length, embedding_dim)
input_tensor = torch.rand(16, 64, dim)

# Forward pass through ResidualDenseSwinBlock
output_tensor = residual_dense_swin_block(input_tensor)

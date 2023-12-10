import torch
import torch.nn as nn


# Shallow Feature Extraction Module
class ShallowFeatureExtractionModule(nn.Module):
    def __init__(self, input_channels, num_filters):
        super(ShallowFeatureExtractionModule, self).__init__()
        self.conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)


class ShiftedWindowAttention(nn.Module):
    def __init__(self, dim, window_size):
        super(ShiftedWindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V):
        B = torch.zeros_like(Q)  # Initialize relative position bias, adjust as needed
        attention = torch.einsum('bhwk,bhwl->bhwkl', Q, K) / (self.dim ** 0.5) + B
        attention = self.softmax(attention)
        output = torch.einsum('bhwkl,bhwl->bhwk', attention, V)
        return output


# Swin Transformer Block

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads=8):
        super(SwinTransformerBlock, self).__init__()

        # Shifted Window Self-Attention
        self.window_attention = ShiftedWindowAttention(dim, window_size)

        # Feedforward layer
        self.feedforward = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=3, stride=1, padding=1)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Residual connection
        residual = x

        # Shifted Window Self-Attention
        attn_output = self.window_attention(x, x, x)
        x = residual + attn_output

        # Layer normalization
        x = self.norm1(x)

        # Feedforward
        ff_output = self.feedforward(x)
        x = x + ff_output

        # Layer normalization
        x = self.norm2(x)

        return x


# Residual Dense Swin Transformer Block (RDSTB)
class RDSTB(nn.Module):
    def __init__(self, input_channels, swin_dim, num_blocks, num_filters):
        super(RDSTB, self).__init__()

        # Swin Transformer and Convolutional Layer
        self.swin_conv_block = nn.Sequential(
            SwinTransformerBlock(input_channels, swin_dim),
            nn.Conv2d(swin_dim, num_filters, kernel_size=3, stride=1, padding=1)
        )

        # Densely Connected Residual Blocks
        self.dense_blocks = nn.ModuleList([SwinTransformerBlock(num_filters, swin_dim) for _ in range(num_blocks)])

    def forward(self, x):
        features = self.swin_conv_block(x)

        for block in self.dense_blocks:
            features = block(features)

        return features


# Upsampling Module
class UpsamplingModule(nn.Module):
    def __init__(self, input_channels):
        super(UpsamplingModule, self).__init__()
        self.upsample_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.upsample_conv(x)


# Generator
class Generator(nn.Module):
    def __init__(self, input_channels, num_filters, swin_dim, num_blocks):
        super(Generator, self).__init__()

        # Shallow Feature Extraction Module
        self.shallow_feature_extraction = ShallowFeatureExtractionModule(input_channels, num_filters)

        # Deep Feature Extraction Module (RDSTB)
        self.deep_feature_extraction = RDSTB(num_filters, swin_dim, num_blocks, num_filters)

        # Upsampling Module
        self.upsampling_module = nn.Sequential(
            UpsamplingModule(num_filters),
            UpsamplingModule(num_filters)
        )

    def forward(self, x):
        # Shallow Feature Extraction
        shallow_features = self.shallow_feature_extraction(x)

        # Deep Feature Extraction
        deep_features = self.deep_feature_extraction(shallow_features)

        # Upsampling Module
        upsampled_features = self.upsampling_module(deep_features)

        # Aggregate shallow and deep features
        generated_image = upsampled_features + shallow_features

        return generated_image


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, output_channels=2, hidden_dim=64):
        super(Discriminator, self).__init__()

        # Feature Extraction Module (Simplified Swin Transformer)
        self.feature_extraction = nn.Sequential(
            SwinTransformerBlock(input_channels, hidden_dim),
            SwinTransformerBlock(hidden_dim, hidden_dim * 2)
        )

        # Classification Head
        self.classification_head = nn.Linear(hidden_dim * 2, output_channels)

    def forward(self, x):
        # Feature Extraction
        features = self.feature_extraction(x)

        # Global Average Pooling
        features = torch.mean(features, dim=[2, 3])

        # Classification Head
        logits = self.classification_head(features)

        return logits

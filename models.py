import torch
import torch.nn as nn


class GeneratorLoss(nn.Module):
    def __init__(self, lambda_adv):
        super(GeneratorLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.criterion_L1 = nn.L1Loss()

    def forward(self, xr, zf, discriminator_real, discriminator_fake):
        # Content Loss (L1 Pixel Loss)
        Lcont = self.criterion_L1(xr, zf)

        # Adversarial Loss for Generator (Relativistic Average GAN)
        adv_loss_real = -torch.log(1 - discriminator_real(xr)).mean()
        adv_loss_fake = -torch.log(discriminator_fake(zf)).mean()
        Ladv = adv_loss_real + adv_loss_fake

        # Total Generator Loss
        LG = Lcont + self.lambda_adv * Ladv

        return LG


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, xr, zf, discriminator_real, discriminator_fake):
        # Adversarial Loss for Discriminator (Relativistic Average GAN)
        adv_loss_real = -torch.log(1 - discriminator_real(xr)).mean()
        adv_loss_fake = -torch.log(1 - discriminator_fake(zf)).mean()
        LD = adv_loss_real + adv_loss_fake

        return LD


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=window_size, stride=window_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)

        # Reshape for self-attention
        x = x.view(B, 2, self.num_heads, -1, H, W)
        x = x.permute(0, 2, 1, 3, 4, 5)

        # Reshape for self-attention computation
        x = x.view(B * self.num_heads, 2, -1, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        return x


class STB(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(STB, self).__init__()

        self.window_attn = WindowAttention(dim, window_size, num_heads)
        self.norm1 = nn.LayerNorm([num_heads, dim, window_size, window_size])

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * num_heads, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
        )
        self.norm2 = nn.LayerNorm([num_heads, dim, window_size, window_size])

    def forward(self, x):
        # Window-based self-attention
        attn = self.window_attn(x)
        x = x + attn
        x = self.norm1(x)

        # MLP
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels, patch_size, embedding_dim, num_heads, window_size, image_size=256):
        super(Discriminator, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = image_size // patch_size
        self.embedding_dim = embedding_dim

        # Convolutional layers for patch discrimination
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        )

        # Linear embedding layer
        self.embedding = nn.Linear(self.num_patches * self.num_patches, embedding_dim)

        # Swin Transformer block 1
        self.swin_block1 = STB(embedding_dim, num_heads, window_size)

        # Upsample layer (transpose convolution) for patch merging
        self.upsample = nn.ConvTranspose2d(embedding_dim, 1, kernel_size=4, stride=2, padding=1)

        # Swin Transformer block 2
        self.swin_block2 = STB(1, num_heads,
                               window_size)  # Adjust input channels for the output of upsampling

        # Classification head
        self.classification_head = nn.Linear(1, 1)

        # Sigmoid activation at the end
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Patch discrimination
        x = self.conv_blocks(x)

        # Reshape to (batch size, num_patches * num_patches)
        x = x.view(-1, self.num_patches * self.num_patches)

        # Linear embedding
        x = self.embedding(x)

        # Reshape to (batch size, num_patches, embedding_dim)
        x = x.view(-1, self.num_patches, self.embedding_dim)

        # Apply Swin Transformer block 1
        x = self.swin_block1(x)

        # Upsample for patch merging
        x = self.upsample(x)

        # Apply Swin Transformer block 2
        x = self.swin_block2(x)

        # Global average pooling
        x = x.mean(dim=(2, 3))

        # Classification head
        x = self.classification_head(x)

        # Apply sigmoid activation
        x = self.sigmoid(x)

        return x


class Generator(nn.Module):
    def __init__(self, input_size, output_channels, image_size=64):
        super(Generator, self).__init__()

        self.image_size = image_size

        # Shallow feature extraction
        self.shallow_feature_extraction = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Fully connected layer to start the generation
        self.fc = nn.Linear(input_size, 256 * (image_size // 8) * (image_size // 8))
        self.relu = nn.ReLU()

        # ResidualDenseSwinBlock repeated three times
        self.rdstb1 = RDSTB(256, heads=8, window_size=8)
        self.rdstb2 = RDSTB(256, heads=8, window_size=8)
        self.rdstb3 = RDSTB(256, heads=8, window_size=8)

        # Upsampling layers
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

        # Convolutional layer to produce the final output
        self.conv_out = nn.Conv2d(256, output_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # Linear layer and unflatten
        self.linear_unflatten = nn.Sequential(
            nn.Linear(output_channels * (image_size ** 2), output_channels * image_size * image_size),
            nn.Unflatten(1, (output_channels, image_size, image_size))
        )

    def forward(self, x):
        # Shallow feature extraction
        x = self.shallow_feature_extraction(x)

        x = self.fc(x.view(x.size(0), -1))
        x = self.relu(x)
        x = x.view(-1, 256, self.image_size // 8, self.image_size // 8)  # Reshape

        # Apply ResidualDenseSwinBlocks
        x = self.rdstb1(x)
        x = self.upsample1(x)
        x = self.rdstb2(x)
        x = self.upsample2(x)
        x = self.rdstb3(x)
        x = self.upsample3(x)  # 4x UpScaling

        # Final convolution and activation
        x = self.conv_out(x)
        x = self.tanh(x)

        # Linear layer and unflatten
        x = self.linear_unflatten(x.view(x.size(0), -1))

        return x


class RDSTB(nn.Module):
    def __init__(self, dim, heads, window_size, mlp_ratio=4.0, num_dense_blocks=3):
        super(RDSTB, self).__init__()

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

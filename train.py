import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Generator, Discriminator, GeneratorLoss, DiscriminatorLoss

# Example usage:
latent_dim = 100
num_residual_blocks = 6
generator = Generator(latent_dim, num_residual_blocks)
discriminator = Discriminator(input_channels=3, patch_size=64, embedding_dim=256, num_heads=8, window_size=7,
                              image_size=256)
generator_loss = GeneratorLoss(lambda_adv=0.01)
discriminator_loss = DiscriminatorLoss()

# Define your dataset and dataloaders
# For this example, let's use the CIFAR-10 dataset
transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Define optimizers for Generator and Discriminator
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)

        # Train Discriminator
        optimizer_D.zero_grad()

        # Generate fake images
        fake_images = generator(torch.randn_like(real_images).to(device))

        # Forward pass through the discriminator for real and fake images
        discriminator_real = discriminator(real_images)
        discriminator_fake = discriminator(fake_images.detach())

        # Calculate Discriminator Loss
        loss_D = discriminator_loss(real_images, fake_images, discriminator_real, discriminator_fake)

        # Backward and optimize Discriminator
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        # Forward pass through the discriminator for real and fake images
        discriminator_real = discriminator(real_images)
        discriminator_fake = discriminator(fake_images)

        # Calculate Generator Loss
        loss_G = generator_loss(real_images, fake_images, discriminator_real, discriminator_fake)

        # Backward and optimize Generator
        loss_G.backward()
        optimizer_G.step()

        # Print training statistics
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                  f"Discriminator Loss: {loss_D.item():.4f}, Generator Loss: {loss_G.item():.4f}")

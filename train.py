import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Generator, Discriminator

# Create instances of Generator and Discriminator
generator = Generator(input_size=100, output_channels=3, image_size=64)
discriminator = Discriminator(input_channels=3, patch_size=64, embedding_dim=256, num_heads=8, window_size=7,
                              image_size=256)

# Define your dataset and dataloaders
# For this example, let's use the CIFAR-10 dataset
transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Define loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for GANs
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

        # Forward pass real images through discriminator
        real_labels = torch.ones((real_images.size(0), 1)).to(device)
        real_outputs = discriminator(real_images)
        loss_real = criterion(real_outputs, real_labels)

        # Forward pass fake images through discriminator
        fake_labels = torch.zeros((real_images.size(0), 1)).to(device)
        fake_outputs = discriminator(fake_images.detach())
        loss_fake = criterion(fake_outputs, fake_labels)

        # Total discriminator loss
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        # Forward pass fake images through discriminator
        fake_outputs = discriminator(fake_images)
        loss_G = criterion(fake_outputs, real_labels)

        # Generator loss
        loss_G.backward()
        optimizer_G.step()

        # Print training statistics
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                  f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

# Save your trained models if needed
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

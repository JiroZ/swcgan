import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models2 import Generator, Discriminator

# Define your generator and discriminator models
# Make sure to replace GeneratorModel and DiscriminatorModel with your actual model classes
generator = Generator(3, 64, 256, 12)
discriminator = Discriminator(3,2, 64 )

# Define the loss functions
criterion_generator = nn.L1Loss()
criterion_adversarial = nn.BCEWithLogitsLoss()

# Define optimizer for both generator and discriminator
optimizer_generator = optim.Adam(generator.parameters(), lr=0.001)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.001)

# Assuming you have a custom dataset class, replace CustomDataset with your actual dataset class
dataloader = DataLoader(CustomDataset(...), batch_size=..., shuffle=True, num_workers=...)

# Training loop
num_epochs = 100
lambda_adv = 0.1  # Adjust this hyperparameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator.to(device)
discriminator.to(device)

for epoch in range(num_epochs):
    for real_images, low_resolution_images in dataloader:
        real_images, low_resolution_images = real_images.to(device), low_resolution_images.to(device)

        # Discriminator update
        optimizer_discriminator.zero_grad()

        # Generate high-resolution images from the low-resolution ones
        generated_images = generator(low_resolution_images)

        # Adversarial loss for real images
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        loss_real = criterion_adversarial(discriminator(real_images), real_labels)

        # Adversarial loss for fake images
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)
        loss_fake = criterion_adversarial(discriminator(generated_images.detach()), fake_labels)

        # Total discriminator loss
        loss_discriminator = loss_real + loss_fake
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Generator update
        optimizer_generator.zero_grad()

        # Content loss
        loss_content = criterion_generator(generated_images, real_images)

        # Adversarial loss for generator
        adversarial_output = discriminator(generated_images)
        loss_adversarial = criterion_adversarial(adversarial_output, real_labels)

        # Total generator loss
        loss_generator = loss_content + lambda_adv * loss_adversarial
        loss_generator.backward()
        optimizer_generator.step()

    # Print training progress (optional)
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Generator Loss: {loss_generator.item():.4f}, "
          f"Discriminator Loss: {loss_discriminator.item():.4f}")

# Save or use the trained generator model as needed
torch.save(generator.state_dict(), 'generator_model.pth')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.gan_model import Generator, Discriminator
from src.data_preparation import prepare_data

def train_gan(num_epochs=100, batch_size=64, learning_rate=0.0002):
    X_train, _, _, _ = prepare_data()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    # Initialize the networks
    input_dim = X_train.shape[1]
    generator = Generator(input_dim=100, output_dim=input_dim)
    discriminator = Discriminator(input_dim=input_dim)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for i in range(0, X_train_tensor.size(0), batch_size):
            real_data = X_train_tensor[i:i+batch_size]
            batch_size = real_data.size(0)

            # Train discriminator
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            outputs = discriminator(real_data)
            d_loss_real = criterion(outputs, real_labels)

            noise = torch.randn(batch_size, 100)
            fake_data = generator(noise)
            outputs = discriminator(fake_data.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train generator
            noise = torch.randn(batch_size, 100)
            fake_data = generator(noise)
            outputs = discriminator(fake_data)
            g_loss = criterion(outputs, real_labels)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    return generator, discriminator
import torch

def generate_synthetic_data(generator, num_samples):
    noise = torch.randn(num_samples, 100)
    synthetic_data = generator(noise)
    return synthetic_data
import torch

# Set device (CPU or GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 28 * 28  # MNIST image size is 28x28
hidden_size = 128
num_classes = 10
batch_size = 64
learning_rate = 0.001
num_epochs = 3


from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from option import batch_size

# Step2: define dataset #
# Load MNIST dataset
train_dataset = MNIST(root=".", train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root=".", train=False, download=True, transform=ToTensor())
# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

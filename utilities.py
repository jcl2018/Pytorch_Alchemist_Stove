import torch.optim as optim
from model import mlp_model
from option import learning_rate

adam_optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)
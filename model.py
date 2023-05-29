import torch.nn as nn

from option import input_size, hidden_size, num_classes, device
from abc import ABC, abstractmethod


# Define the interface
class ModelInterface(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def print_model(self):
        pass


class LayerInterface(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

class DenseLayer(LayerInterface):
    def __init__(self, input_size, output_size, activation=None):
        super(LayerInterface, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.act = activation

    def forward(self, x):
        x = self.fc(x)
        if self.act:
            x = self.act(x)
        return x


class MLPModel(ModelInterface):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ModelInterface, self).__init__()

        self.flatten = nn.Flatten().to(device)
        self.dense1 = DenseLayer(input_size, hidden_size, activation=nn.ReLU()).to(device)
        self.dense2 = DenseLayer(hidden_size, num_classes).to(device)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def print_model(self):
        print(self.dense1)
        print(self.dense2)


# Initialize the model
mlp_model = MLPModel(input_size, hidden_size, num_classes)
mlp_model.print_model()

import torch.nn as nn
import torch.optim as optim
import torch

# creates a neuronetwork
class mlp_voltage_predictor(nn.Module):
    # sets up layers (2) and activation function (ReLU)
    def __init__(self, input_size, output_size):
        super().__init__()
        #self.layer1 = nn.Linear(input_size, 32)
        self.layer1 = nn.LSTM(input_size, hidden_size = 32, num_layers=1)
        #self.activationFunction = nn.ReLU()
        self.layer2 = nn.Linear(32, 8)
        self.activationFunction2 = nn.ReLU()
        self.layer3 = nn.Linear(8, output_size)

    # sets how data moves thorugh layers
    def forward(self, x):
        x, _=self.layer1(x) # input goes through first layer
        #x = self.activationFunction(x) # ReLU is activated
        print(x.size())
        x = self.layer2(x) # data goes through second layer
        x = self.activationFunction2(x) # ReLU is activated 

        x = self.layer3(x)
        x = self.activationFunction2(x)
        return x # output is returned.
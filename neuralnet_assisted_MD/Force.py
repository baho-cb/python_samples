import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary


class TheNet(nn.Module):
    """basic FF network for approximating functions"""
    def __init__(self, nn_width, num_hidden,input_size):
        super().__init__()

        self.layer_first = nn.Linear(input_size, nn_width)

        layers = []
        for _ in range(num_hidden):
            layers.append(nn.Linear(nn_width, nn_width))
        self.layer_hidden = nn.ModuleList(layers)

        self.layer_last = nn.Linear(nn_width, 1)

    def forward(self, x):
        activation = nn.ReLU()
        u = activation(self.layer_first(x))
        for hidden in self.layer_hidden:
            u = activation(hidden(u))
        u = self.layer_last(u)
        return u

class TheSelectorNet(nn.Module):
    """basic FF network for approximating functions"""
    def __init__(self, nn_width, num_hidden,input_size):
        super().__init__()

        self.layer_first = nn.Linear(input_size, nn_width)

        layers = []
        for _ in range(num_hidden):
            layers.append(nn.Linear(nn_width, nn_width))
        self.layer_hidden = nn.ModuleList(layers)

        self.layer_last = nn.Linear(nn_width, 2)

    def forward(self, x):
        activation = nn.ReLU()
        u = activation(self.layer_first(x))
        for hidden in self.layer_hidden:
            u = activation(hidden(u))
        u = self.layer_last(u)
        return u

class TheEnergyNet(nn.Module):
    """basic FF network for approximating functions"""
    def __init__(self, nn_width, num_hidden,input_size):
        super().__init__()

        self.layer_first = nn.Linear(input_size, nn_width)

        layers = []
        for _ in range(num_hidden):
            layers.append(nn.Linear(nn_width, nn_width))
        self.layer_hidden = nn.ModuleList(layers)

        self.layer_last = nn.Linear(nn_width, 1)

    def forward(self, x):
        activation = nn.ReLU()
        u = activation(self.layer_first(x))
        for hidden in self.layer_hidden:
            u = activation(hidden(u))
        u = self.layer_last(u)
        return u


class Force():

    def __init__(self,root_name):
        """
        Force components are calculated by three neural nets
        root_name : force_NN_model
        saved torch models : force_NN_model1.pth, force_NN_model2.pth, force_NN_model3.pth
        """
        self.NN_paths = []
        # self.W = 110
        # self.D = 15
        self.input_size = 6
        # for i in range(1,7):
        #     s = root_name + str(i) + '.pth'
        #     self.NN_paths.append(s)
        self.read_NN()

    def read_NN(self):
        # self.NN_models = []
        # for path in self.NN_paths:
        #     model = TheNet(self.W,self.D,self.input_size)
        #     model.load_state_dict(torch.load(path,map_location='cpu'))
        #     model.to('cpu')
        #     model.eval()
        #     self.NN_models.append(model)

        selector_model =  TheSelectorNet(70,5,self.input_size)
        selector_model.load_state_dict(torch.load('models/model_selector.pth',map_location='cpu'))
        selector_model.to('cpu')
        selector_model.eval()
        self.selector = selector_model

        energy_model =  TheEnergyNet(80,10,self.input_size)
        energy_model.load_state_dict(torch.load('models/model_energy.pth',map_location='cpu'))
        energy_model.to('cpu')
        energy_model.eval()
        self.energy_net = energy_model





























    def dummy(self):
        pass

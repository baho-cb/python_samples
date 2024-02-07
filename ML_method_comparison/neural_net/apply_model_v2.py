import torch
import torch.nn as nn
import numpy as np
import sys,os
from torch.utils.data import DataLoader, Dataset,random_split
import time
import argparse
import re
import gc
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, threshold=None, suppress=True)
torch.set_printoptions(precision=6, threshold=None)

"""
April 11

Takes x.npy and the model just to reproduce the training output

"""


class MyDataset(Dataset):
    def __init__(self,filenames):
        self.x_data = torch.load(filenames[0])
        self.y_data = torch.load(filenames[1])
        shape = self.x_data.size()
        self.n_samples = shape[0]
        self.y_data = torch.reshape(self.y_data,(self.n_samples,1))
        self.input_size = shape[1]

    def get_min_target(self):
        """
        to calculate relative error I must get rid of values zeros
        since they dominate the mean rel err so just when calculate
        rel err to print I add the minimum + 1.0 to all target and prediction
        """
        return torch.min(self.y_data).numpy()

    def normalize_input(self):
        """
        Max-min normalization
        """
        N_features = self.x_data.size()[1]
        for i in range(N_features):
            data = torch.clone(self.x_data[:,i])
            min = torch.min(data)
            max = torch.max(data)
            data_normal = (data - min) / (max - min)
            self.x_data[:,i] = data_normal


    def load_to_gpu(self,device):
        self.x_data = self.x_data.to(device)
        self.y_data = self.y_data.to(device)

    def get_split_counts(self,fraction):
        N_test = self.n_samples*fraction
        N_test = int(N_test)
        N_train = self.n_samples - N_test
        return N_train,N_test

    def __getitem__(self,index):
        return self.x_data[index,:],self.y_data[index]

    def __len__(self):
        return self.n_samples

    def get_input_size(self):
        return self.input_size


class TheNet(nn.Module):
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


def get_network_structure(model_path):
    model_str = re.split("_",model_path)
    width = 0
    depth = 0
    for mm in model_str:
        if(len(mm)>1):
            if(mm[0]=='W' and mm[1:].isdigit()):
                width = int(mm[1:])
            if(mm[0]=='D' and mm[1:].isdigit()):
                depth = int(mm[1:])

    if(width < 1 or depth < 1):
        print("Error 34TH")
        # exit()

    # print("WARNING !!! WIDTH AND DEPTH ARE HARDCODED !!! WARNING ")
    # width = 85
    # depth = 13
    return width,depth



parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_datas", nargs='+', required=True, help="-i x.pt" )
non_opt.add_argument('--model', metavar="<dat>", type=str, dest="model_path", required=True, help="-" )

args = parser.parse_args()
input_datas = args.input_datas
model_path = args.model_path

if not os.path.exists('./raw_predictions/'):
    os.makedirs('./raw_predictions/')

print("Reading x data ...")
x_data = np.load(input_datas[0])
x_data = torch.from_numpy(x_data)

# x_data = torch.load(input_datas[0],map_location=torch.device('cpu'))


print("Min max normalization ...")

"""minmax normalize xdata"""
N_features = x_data.size()[1]

"""
Since we now use the tree for pre-selection of the finite data we must use the same
min max values that we have trained data with and it can be different than what is
used in the simulator script
"""
# print("LOOK HERE FIRST !!!! ")
# exit()
mins = [0.0,0.0,0.0,0.0]
maxs = [7.00,7.00,3.141593,3.141593]

for i in range(N_features):
    data = torch.clone(x_data[:,i])
    # min = torch.min(data)
    # max = torch.max(data)
    min = mins[i]
    max = maxs[i]
    # print(min,max)
    data_normal = (data - min) / (max - min)
    x_data[:,i] = data_normal

del(data)
del(data_normal)

print("Reading Model ...")
N_divide = 60
x_data = torch.split(x_data,x_data.size()[0]//N_divide)
y_pred = torch.tensor([[0.0,0.0],[0.0,0.0]])
width,depth = get_network_structure(model_path)
model = TheNet(width,depth,x_data[0].size()[1])
model.load_state_dict(torch.load(model_path))
model.to('cpu')
model.eval()
print("Applying model ...")
for i,x in enumerate(x_data):
    print("%d/%d"%(i,N_divide))
    y = model(x)
    y_pred = torch.vstack((y_pred,y.detach()))
y_pred = y_pred[2:]
print("Saving predictions ...")
torch.save(y_pred,'./raw_predictions/y_pred_%s.pt'%(model_path[9:]))
print("DONE")

exit()

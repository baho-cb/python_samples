import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import sys,os
import re

def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_network_structure(model_path):
    model_str = re.split("_",model_path)
    width = 0
    depth = 0
    Nrate = 0
    epoch = 0
    for mm in model_str:
        if(len(mm)>1):
            if(mm[0]=='W' and mm[1:].isdigit()):
                width = int(mm[1:])
            if(mm[0]=='D' and mm[1:].isdigit()):
                depth = int(mm[1:])
            if(mm[:5]=="Nrate" and isfloat(mm[5:])):
                Nrate = float(mm[5:])
            if(mm[:2]=="NE" and mm[2:].isdigit()):
                epoch = int(mm[2:])


    if(width < 1 or depth < 1):
        print("Error 34TH")
        # exit()

    # print("WARNING !!! WIDTH AND DEPTH ARE HARDCODED !!! WARNING ")
    # width = 85
    # depth = 13
    return width,depth,Nrate,epoch


if not os.path.exists('./pred_res/'):
    os.makedirs('./pred_res/')

parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('--y_true', metavar="<dat>", type=str, dest="y_true", required=True, help="-i y_pred_raw.pt y_true.pt" )
non_opt.add_argument('--y_pred', metavar="<dat>", type=str, dest="y_pred", required=True, help="-i y_pred_raw.pt y_true.pt" )

args = parser.parse_args()
y_pred_str = args.y_pred
y_true_str = args.y_true

width,depth,Nrate,epoch = get_network_structure(y_pred_str)

y_pred = torch.load(y_pred_str)
y_pred_class = 1.0/(1.0+torch.exp(-y_pred))
y_pred_class = torch.argmax(y_pred_class,dim=1)
y_pred_class = y_pred_class.detach().numpy()

y_test = np.load(y_true_str)

y_test_class = np.ones_like(y_test,dtype=np.int32)
y_test_class[np.abs(y_test)<0.05] = 0

err = np.abs(y_pred_class - y_test_class)
err_rate = len(err[err>0.01])/len(err)

print("%d %d %d %.2f %.5f"%(depth,width,epoch,Nrate,err_rate))



exit()

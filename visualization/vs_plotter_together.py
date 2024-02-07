import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import sys,os
import re
import matplotlib as mpl

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['text.usetex'] = True

cm = 1/2.54
mpl.rcParams['savefig.bbox'] = "tight"
mpl.rcParams['savefig.pad_inches'] = 0.1
mpl.rcParams['savefig.transparent'] = True
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.shadow'] = False


# parser = argparse.ArgumentParser(description="")
# non_opt = parser.add_argument_group("mandatory arguments")
# non_opt.add_argument('--y_true', metavar="<dat>", type=str, dest="y_true", required=True, help="-i y_pred_raw.pt y_true.pt" )
# non_opt.add_argument('--y_pred', metavar="<dat>", type=str, dest="y_pred", required=True, help="-i y_pred_raw.pt y_true.pt" )
# non_opt.add_argument('--power', metavar="<float>", type=float, dest="power", required=True, help="-i x.pt y.pt" )

# args = parser.parse_args()
# y_pred_str = args.y_pred
# y_true_str = args.y_true
# power = args.power

y_pred_str = 'y_pred_del0_NE90.pth.pt'
y_pred_cube_str = 'y_pred_del1_NE160_.pth.pt'

y_true_str = 'merged_test_int_cyl.npy'
y_true_cube_str = 'merged_test_int_cube.npy'

fsize = 30
xx = np.linspace(-20.70,15.0,num=1000)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(9.6, 4.8))

######### CYLINDER ##############
col = -1 # 1-9
splitted = re.split("_",y_pred_str)
for split in splitted:
    if(len(split)>3):
        if(split[:3]=='del'):
            col = int(split[3])

mins = np.array([-26.44549,-5.195254,-12.5106,-6.401458,-35.7029,-0.692])
maxs = np.array([6.65013,5.181416,2.74002,6.406832,6.061378,0.69235])

supplied_min = mins[col]
supplied_max = maxs[col]

y_pred = torch.load(y_pred_str)
y_pred_raw = y_pred.detach().numpy()
y_pred_raw = y_pred_raw.flatten()

y_true = np.load(y_true_str)
y_true = y_true[:,col+1] ## energy is col 0

y_pred_true_range = y_pred_raw*(supplied_max-supplied_min) + supplied_min
y_true_minmaxed = (y_true - supplied_min) / (supplied_max - supplied_min)

ax1.scatter(y_true[:1500],y_pred_true_range[:1500],s=7.0,marker='o')
# plt.plot(xx,xx,'k')
ax1.plot(xx,xx*0.9,'k--',linewidth=2.0,label='10 \% error line',zorder=-1)
ax1.plot(xx,xx*1.1,'k--',linewidth=2.0,zorder=-1)
lll =6.5
ax1.set_xlim(-lll,lll)
ax1.set_ylim(-lll,lll)
ax1.set_xlabel("True $F_{x}$",fontsize=fsize, labelpad=20)
ax1.set_ylabel("Predicted $F_{x}$",fontsize=fsize, labelpad=20)
ax1.legend(prop={'size': 20})
ax1.tick_params(axis='both', which='major', labelsize=fsize)


########### CYLINDER #################


########### CUBE ####################

col = -1 # 1-9
splitted = re.split("_",y_pred_cube_str)
for split in splitted:
    if(len(split)>3):
        if(split[:3]=='del'):
            col = int(split[3])

mins = np.array([-4.2,-26.00,-9.75,-4.0,-1.50,-13.80,-3.00]) # v4
maxs = np.array([12.50,9.3,2.25,1.25,4.5,6.25,17.50])

supplied_min = mins[col]
supplied_max = maxs[col]

y_pred_cube = torch.load(y_pred_cube_str)
y_pred_raw_cube = y_pred_cube.detach().numpy()
y_pred_raw_cube = y_pred_raw_cube.flatten()

y_true_cube = np.load(y_true_cube_str)
y_true_cube = y_true_cube[:,col] ## energy is col 0

y_pred_true_range_cube = y_pred_raw_cube*(supplied_max-supplied_min) + supplied_min
y_true_minmaxed_cube = (y_true_cube - supplied_min) / (supplied_max - supplied_min)

ax2.scatter(y_true_cube[:1500],y_pred_true_range_cube[:1500],s=7.0,marker='o')
# plt.plot(xx,xx,'k')
ax2.plot(xx,xx*0.9,'k--',linewidth=2.0,zorder=-1)
ax2.plot(xx,xx*1.1,'k--',linewidth=2.0,zorder=-1)
lll =6.5
ax2.set_xlim(-lll,lll)
ax2.set_ylim(-lll,lll)
ax2.set_xlabel("True $F_{x}$",fontsize=fsize, labelpad=20)
# ax2.set_ylabel("Predicted $F_{x}$",fontsize=25, labelpad=20)
# ax2.legend(prop={'size': 20})
ax2.tick_params(axis='both', which='major', labelsize=fsize)
ax2.yaxis.set_visible(False)

########### CUBE ####################
# plt.subplots_adjust(wspace=0)
plt.tight_layout()
# plt.savefig("Fx_parity_v4.pdf")
plt.show()

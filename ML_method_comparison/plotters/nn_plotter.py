import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import argparse
import numpy as np
import re
import matplotlib as mpl
import matplotlib.patches as mpatches

np.set_printoptions(precision=5,suppress=True)

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

"""
# depth,width,epoch,Nrate,err_rate
depth,n_estimators,N_rate,err_rate,tf,tf_inf
# depth,width,Nrate,err_rate
"""




""" needs some preprocessing due to epochs """
# data = np.loadtxt("nn.txt")

# nets = np.unique(data[:,:2],axis=0)
# final_data = np.zeros((27,5))
# ns = np.array([0.05,0.1,0.2])
#
# cc = 0
#
# for i,net in enumerate(nets):
#     for j,n in enumerate(ns):
#         indexes = []
#         errors = []
#         for k,d in enumerate(data):
#             if(d[0]==net[0] and d[1]==net[1] and d[3]==n):
#                 indexes.append(k)
#                 errors.append(d[-1])
#         errors = np.array(errors)
#         max_index = np.argmin(errors)
#         ind = indexes[max_index]
#         final_data[cc] = data[ind]
#         cc += 1
#
# # print(final_data)
# final_data = np.delete(final_data, 2, 1)
# d=final_data
# print(final_data)
# # exit()
# """
# PLOT ERROR RATE
# """
#
# markers=['o','v','x']
# labels=["Training Size = N","Training Size = 2N","Training Size = 4N"]
# sizes=[135,180,225]
# colors=['b','r','g']
# plt.figure(figsize=(8,6))
# for i,size in enumerate(np.unique(d[:,2])):
#     for j,n_estim in enumerate(np.unique(d[:,1])):
#         dat = d[d[:,2]==size]
#         dat = dat[dat[:,1]==n_estim]
#         plt.scatter(dat[:,0],100.0*dat[:,3],marker=markers[i],c=colors[j],s=sizes[i])
#
#
# ss = 150.0
#
# plt.scatter(100.0,100.0,marker=markers[0],label=labels[0],c='k',s=ss)
# plt.scatter(100.0,100.0,marker=markers[1],label=labels[1],c='k',s=ss)
# plt.scatter(100.0,100.0,marker=markers[2],label=labels[2],c='k',s=ss)
# plt.scatter(100.0,100.0,marker='s',label='Width = 30',c=colors[0],s=ss)
# plt.scatter(100.0,100.0,marker='s',label='Width = 50',c=colors[1],s=ss)
# plt.scatter(100.0,100.0,marker='s',label='Width = 70',c=colors[2],s=ss)
#
#
# plt.ylim(0.0,0.9)
# plt.xlim(4.0,10.0)
# plt.xlabel("Depth",fontsize=25, labelpad=20)
# plt.ylabel("Error Rate (\%)",fontsize=25, labelpad=20)
# plt.tick_params(axis='both', which='major', labelsize=30)
# plt.legend(prop={'size': 17},loc="lower right")
# plt.savefig("nn_error_v1.pdf")
# # plt.show()
#
# exit()

"""
PLOT INFERENCE

D W Time
"""

d = np.loadtxt("nn_time.txt")
markers=['o','v','x']
labels=["Training Size = N","Training Size = 2N","Training Size = 4N"]
sizes=[180,225,275]
colors=['b','r','g']
plt.figure(figsize=(8,6))
for j,w in enumerate(np.unique(d[:,1])):
    dat = d[d[:,1]==w]
    # dat = dat[dat[:,2]==0.2]
    plt.scatter(dat[:,0],dat[:,-1],marker='s',c=colors[j],s=180,label="Width=%d"%int(w))


ss = 150.0

plt.xlabel("Depth",fontsize=25, labelpad=20)
plt.ylabel("Inference Time in seconds",fontsize=25, labelpad=20)
plt.tick_params(axis='both', which='major', labelsize=30)
plt.legend(prop={'size': 17})
plt.savefig("nn_inference_v1.pdf")
# plt.show()

exit()
#










exit()

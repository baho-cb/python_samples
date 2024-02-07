import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import argparse
import numpy as np
import re
import matplotlib as mpl
import matplotlib.patches as mpatches

np.set_printoptions(precision=3,suppress=True)

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
# depth,n_estimators,N_rate,err_rate,tf,tf_inf
"""


d = np.loadtxt("rf.txt")


"""
PLOT ERROR RATE
"""

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
# plt.scatter(100.0,100.0,marker='s',label='N_estimators = 10',c=colors[0],s=ss)
# plt.scatter(100.0,100.0,marker='s',label='N_estimators = 25',c=colors[1],s=ss)
# plt.scatter(100.0,100.0,marker='s',label='N_estimators = 50',c=colors[2],s=ss)
#
#
# plt.ylim(0.0,10.5)
# plt.xlim(4.5,15.5)
# plt.xlabel("max depth",fontsize=25, labelpad=20)
# plt.ylabel("Error Rate (\%)",fontsize=25, labelpad=20)
# plt.tick_params(axis='both', which='major', labelsize=20)
# plt.legend(prop={'size': 17})
# # plt.savefig("rf_v1.pdf")
# plt.show()
#
# exit()

"""
PLOT INFERENCE
"""


markers=['o','v','x']
labels=["Training Size = N","Training Size = 2N","Training Size = 4N"]
sizes=[135,180,225]
colors=['b','r','g']
plt.figure(figsize=(8,6))
for j,n_estim in enumerate(np.unique(d[:,1])):
    dat = d[d[:,1]==n_estim]
    dat = dat[dat[:,2]==0.2]
    plt.scatter(dat[:,0],dat[:,-1],marker='s',c=colors[j],s=180,label="N_estimators=%d"%int(n_estim))


ss = 150.0

plt.xlabel("max depth",fontsize=25, labelpad=20)
plt.ylabel("Inference Time in seconds",fontsize=25, labelpad=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend(prop={'size': 17})
plt.savefig("rf_inference.pdf")
# plt.show()

exit()











exit()

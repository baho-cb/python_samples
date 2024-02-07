import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg


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


#### 0.3 kT
rdf_h1 = np.loadtxt("05_hoomd.txt")
rdf_n1 = np.loadtxt("05_nn.txt")

#### 0.5 kT
rdf_h2 = np.loadtxt("06_hoomd.txt")
rdf_n2 = np.loadtxt("06_nn.txt")

#### 0.75 kT
rdf_h3 = np.loadtxt("07_hoomd.txt")
rdf_n3 = np.loadtxt("07_nn.txt")


fs = 13
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8.6*cm,18*cm))

# ax1.scatter(rdf_n1[-1,:],np.average(rdf_n1[:-1,:],axis=0),c='c',marker='o',edgecolors='black',label="NeuralNet",s=40,linewidth=0.5)
# ax1.plot(rdf_h1[-1,:],np.average(rdf_h1[:-1,:],axis=0),c='r',linewidth =2.0,linestyle='--',label="HOOMD",zorder=+1)
ax1.scatter(rdf_n1[-1,:],np.average(rdf_n1[:-1,:],axis=0),c='c',marker='o',edgecolors='black',s=40,linewidth=0.5)
ax1.plot(rdf_h1[-1,:],np.average(rdf_h1[:-1,:],axis=0),c='r',linewidth =2.0,linestyle='--',zorder=+1)
ax1.set_ylabel('g(r)',fontsize=fs, labelpad=10)
# ax1.set_xlabel("$\sigma$",fontsize=fs, labelpad=10)
ax1.tick_params(axis='both', which='major', labelsize=fs)
# plt.ylim(0.0,30)
# ax1.legend()
# ax1.legend(prop={'size': 10})
ax1.set_xlim(2.4,11.7)

ax2.scatter(rdf_n2[-1,:],np.average(rdf_n2[:-1,:],axis=0),c='c',marker='o',label="NeuralNet",edgecolors='black',s=40,linewidth=0.5)
ax2.plot(rdf_h2[-1,:],np.average(rdf_h2[:-1,:],axis=0),c='r',linewidth =2.0,linestyle='--',label="HOOMD",zorder=+1)
ax2.set_ylabel('g(r)',fontsize=fs, labelpad=10)
# ax2.set_xlabel("$\sigma$",fontsize=fs, labelpad=10)
ax2.tick_params(axis='both', which='major', labelsize=fs)
# plt.ylim(0.0,30)
ax2.legend()
ax2.legend(prop={'size': 10})
ax2.set_xlim(2.4,11.7)

ax3.scatter(rdf_n3[-1,:],np.average(rdf_n3[:-1,:],axis=0),c='c',marker='o',edgecolors='black',s=40,linewidth=0.5)
ax3.plot(rdf_h3[-1,:],np.average(rdf_h3[:-1,:],axis=0),c='r',linewidth =2.0,linestyle='--',zorder=+1)
ax3.set_ylabel('g(r)',fontsize=fs, labelpad=10)
ax3.set_xlabel("$\sigma$",fontsize=fs, labelpad=10)
ax3.tick_params(axis='both', which='major', labelsize=fs)
# plt.ylim(0.0,30)
# ax3.legend()
# ax3.legend(prop={'size': 10})
ax3.set_xlim(2.4,11.7)


############ INSETS #################

box1 = [0.45, 0.45, 0.5, 0.5]
ax03 = inset_axes(ax1, width='100%', height='100%', bbox_to_anchor=box1, bbox_transform=ax1.transAxes)
img_b = mpimg.imread('test_ospray.png').astype(np.float32)
# img_b = mpimg.imread('cube_aggregate.png')
ax03.imshow(img_b)
ax03.axis('off')

box2 = [0.60, 0.1, 0.45, 0.45]
ax075 = inset_axes(ax1, width='100%', height='100%', bbox_to_anchor=box2, bbox_transform=ax3.transAxes)
# img_075 = mpimg.imread('kt075.png')
img_075 = mpimg.imread('ospray_kT10.png').astype(np.float32)
ax075.imshow(img_075)
ax075.axis('off')




############ INSETS #################
# plt.tight_layout()


plt.savefig('rdf_cube_subinset_v3.pdf',dpi = 10000)
# plt.show()







exit()

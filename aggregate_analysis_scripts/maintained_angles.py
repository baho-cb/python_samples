#!/home/statt/anaconda3/bin/python
import numpy as np
import sys, os
sys.path.insert(0,"/home/baho/Desktop/progs/hoomd-blue/build/hoomd-2.9.3/")
sys.path.insert(0,"/home/bargun2/Programs/hoomd-blue/build/hoomd-2.9.2/hoomd")
# import hoomd
# import hoomd.md
# import hoomd.mpcd as mpcd
# import gsd.hoomd
from FractalDim import Utils
from ClusterUtils import ClusterUtils
from ClusterNetwork import Cluster
import matplotlib.pyplot as plt
import argparse
import matplotlib as mpl
from scipy.optimize import curve_fit
np.set_printoptions(suppress=True,precision=3)
from statsmodels.graphics import tsaplots
import statsmodels


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
Just stack maintained angles of  aggregates of a shape and dump to a text file for
further analysis
"""

def triplet_check(data):
    """
    Make sure that triplets are in unique order :
    -> On frame 1 you might have 10 30 42, if you have this angle as 42 30 10
    on frame 2 it will be considered as broken since the two triplets are not
    equal.
    ** Same goes for bonds **
    3rd column can't be smaller than 1st
    """
    diff = data[:,2] - data[:,0]
    ind = np.where(diff<0.0)
    ind = ind[0]
    if(len(ind)>0):
        print("Error : CMYLMZ")
        exit()

def autocorr(x):
    x -= np.mean(x)
    result = np.correlate(x, x, mode='full')
    results = result[result.size//2+1:]
    results = results/np.max(results)
    # results = results/results[0]
    return results


def get_maintained_angles(triplets_list,angles_list,N_step):
    """
    From the list of bond groups calculate remaining bond fraction for a trajectory
    https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays

    returns the values fr maintained angles in the first N_step of the simulation
    The challenge is to keep the angles ordered so that ACF can be calculated

    """
    reference_triplets = triplets_list[0]
    for ii,triplets in enumerate(triplets_list[:N_step]):
        nrows, ncols = reference_triplets.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)], 'formats':ncols * [reference_triplets.dtype]}

        C = np.intersect1d(reference_triplets.view(dtype), triplets.view(dtype))
        reference_triplets = np.zeros((len(C),3),dtype=int)
        for i in range(len(C)):
            reference_triplets[i,0]=C[i][0]
            reference_triplets[i,1]=C[i][1]
            reference_triplets[i,2]=C[i][2]


    maintained_triplets = reference_triplets.tolist()

    index_list_of_maintained_angles = []
    for ii,triplets in enumerate(triplets_list[:N_step]):
        c_indexes = []
        triplet_list = triplets.tolist()
        for maintained_triplet in maintained_triplets:
            maintained_tripletl = list(maintained_triplet)
            if(maintained_tripletl[0]>-0.5):
                ind = triplet_list.index(maintained_tripletl)
                c_indexes.append(ind)
        index_list_of_maintained_angles.append(c_indexes)

    # for k in index_list_of_maintained_angles:
    #     print(len(k))

    final_angle_data = np.zeros((N_step,len(index_list_of_maintained_angles[0])))
    for j,indexes in enumerate(index_list_of_maintained_angles):
        final_angle_data[j,:] = angles_list[j][indexes]

    return final_angle_data

def autocorr_angles(angles):
    N_Ts, N_angle = angles.shape
    acf_list = []
    for i in range(N_angle):
        c_angle = angles[:,i]
        acf = autocorr(c_angle)
        acf_list.append(acf)

    plt.figure(1)
    for i,acf in enumerate(acf_list):
        plt.plot(np.arange(len(acf)),acf,c='k',linewidth=0.2)
    # plt.title('ACF for every single maintained angle')
    # plt.title('ACF for every single maintained angle - mean subtracted - normalized by angle[0]')
    plt.title('ACF for every single maintained angle - mean subtracted - normalized by max(angle)')
    plt.show()


    acf = np.array(acf_list)
    acf_avg = np.average(acf,axis = 0)
    acf_std = np.std(acf,axis = 0)
    plt.figure(2)
    plt.plot(np.arange(len(acf_avg)),acf_avg,c='k',linewidth=0.2)
    plt.show()
    exit()


def autocorrFFT(x):
    R"""
    caclulates autocorrelation function of signal

    args: x (array) signal
    returns: autocorrelation function (array)

    There are different convetions to define autocorrelation,
    convention A is the wikipedia one <https://en.wikipedia.org/wiki/Autocorrelation>
    convention B is multiplied by (N-m)

    The Wiener Khinchin theorem says that the  power spectral density (PSD)
    of a function is the Fourier transform of the autocorrelation. See
    <https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem>

    So, compute PSD of x and back fourier transform it to get the
    autocorrelation (in convention B). It's the cyclic autocorrelation, so
    zero-padding is needed to get the non-cyclic autocorrelation.

    """
    N=len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   #now we have the autocorrelation in convention B
    return res
    # n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    # return res/n #this is the autocorrelation in convention A

def plot_angle_profile(angles,filename):

    N_Ts, N_angle = angles.shape
    plt.figure(1)
    for i in range(N_angle):
        plt.plot(np.arange(N_Ts),angles[:,i]*(180.0/np.pi),c='k',linewidth=0.2)
    plt.ylabel('Angle in degrees')
    plt.xlabel('Timesteps')
    # plt.title('%s - %d angles that are maintained '%(filename,N_angle) )
    plt.show()

    plt.figure(2)
    for i in range(N_angle):
        plt.plot(np.arange(N_Ts),angles[:,i]*(1.0/angles[0,i]),c='k',linewidth=0.2)
    plt.ylabel('Angle in degrees')
    plt.xlabel('Timesteps')
    # plt.title('%s - %d angles that are maintained - relative to start '%(filename,N_angle) )
    plt.show()

    exit()

def plot_variance1(angles):
    angles = angles/angles[0,:]
    std = np.std(angles,axis=1)

    plt.figure(1)
    plt.plot(np.arange(len(std)),std)
    plt.show()

parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str,nargs="+", dest="input_files",
required=True, help="take triplets file as input - angle files have the same names anyways " )

args = parser.parse_args()
input_files = args.input_files

"""
I don't know how to use all of the bonds to do this so I will only pick the ones
that are maintained till a given time step and ACF will only consider those angles
from Ts=0. I guess it is possible to include all the angles and simply not considering
them in further timesteps but then I can use available functions.

So first try with kept bonds till timestep then do the better one by hand"""
stop_ts = 200

datas = []
for jj,input_file in enumerate(input_files):
    data_triplets = np.loadtxt(input_file,dtype=int)
    angle_file = input_file[:-8] + "angles"
    data_angles = np.loadtxt(angle_file,dtype=float)
    triplet_check(data_triplets)

    indexes = np.where(data_triplets[:,0]==-1)
    indexes = indexes[0]
    c_triplet_groups = np.split(data_triplets,indexes)
    c_angle_groups = np.split(data_angles,indexes)
    del c_triplet_groups[0]
    del c_angle_groups[0]
    final_angle_data = get_maintained_angles(c_triplet_groups,c_angle_groups,stop_ts)
    datas.append(final_angle_data)

arr = datas[0]
for i in range(1,len(datas)):
    arr = np.hstack((arr,datas[i]))

np.savetxt('sphere134_200_maintained_angles.txt',arr,fmt="%.3f")














exit()

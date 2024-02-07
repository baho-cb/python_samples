import numpy as np
import gsd.hoomd
import os.path
import sys
import itertools
import time
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance
from AnalyzerUtils import Analyzer,wrap_pbc
import string
import freud
import argparse
import matplotlib.pyplot as plt

"""
April 17 - 2023

Calculate Mean Square Displacement of rigid bodies in MD simulation

v1 can only calculate msd for hoomd but for NN simulations we don't have images
dumped out so we need to calculate manually

python3 msd_analyzer_v1.py -i simulation_output.gsd
"""

parser = argparse.ArgumentParser(description="msd analysis")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_files",
required=True, nargs="+", help=".gsd files ")

args = parser.parse_args()
input_files = args.input_files

msds = []
msds_freud = []
for i,input_file in enumerate(input_files):
    in_file=gsd.fl.GSDFile(name=input_file, mode='r',application="hoomd",schema="hoomd", schema_version=[1,0])
    N_frames = in_file.nframes
    frames = np.arange(N_frames)

    energy_data = np.zeros((N_frames,6))
    frame_data = Analyzer(input_file,0)
    N_shapes = frame_data.getNshapes()
    Lx = frame_data.getBoxSize()

    all_central_pos = np.zeros((N_frames,N_shapes,3))
    all_central_images = np.zeros((N_frames,N_shapes,3))

    for i_f,target_frame in enumerate(frames):
        frame_data = Analyzer(input_file,target_frame)
        pos_central_frame = frame_data.getCentralPos()
        image_central_frame = frame_data.getCentralImages()
        all_central_pos[i_f,:,:] = pos_central_frame
        all_central_images[i_f,:,:] = image_central_frame

    """ MANUALLY (matches with Freud)"""

    diff = np.diff(all_central_pos,axis=0)
    diff = wrap_pbc(diff,Lx)
    cumsum = np.cumsum(diff, axis=0)
    cumsum = cumsum**2
    cumsum = np.sum(cumsum,axis=2)
    msd = np.average(cumsum,axis=1)
    msds.append(msd)
    outname = input_file[:-4] + '_msd_direct.txt'
    np.savetxt(outname,msd,fmt="%.5f")



















exit()

import numpy as np
# import gsd.hoomd
import gsd.pygsd
# from hoomd.data import make_snapshot, boxdim
import os.path
import sys
import itertools
import time
from scipy.spatial import cKDTree as KDTree
# from ExtractDataUtils_V2 import ConvertDataUtils
from scipy.spatial import distance
from AnalyzerUtils import Analyzer
import string
import freud
import argparse
import matplotlib.pyplot as plt
"""
April 17

Takes in the gsd and calculates and outputs the pair correlation function of the
center of masses of the rigid bodies

Run as : python3 rdf_analyze.py -i simulation_output.gsd --start 2 --end 5 --L 47.6 
"""

parser = argparse.ArgumentParser(description="rdf analysis")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_files", nargs="+",
required=True, help=".gsd files " )
non_opt.add_argument('--start', metavar="<int>", type=int, dest="start_frame",
required=True, help="start to average from this frame" )
non_opt.add_argument('--end', metavar="<int>", type=int, dest="end_frame",
required=True, help="average up to this frame" )
non_opt.add_argument('--L', metavar="<float>", type=float, dest="box_size",
required=True, help="box size" )

args = parser.parse_args()
input_files = args.input_files
start_frame = args.start_frame
end_frame = args.end_frame
box_size = args.box_size

bins = 150
r_max = 12
rdf_all = np.zeros((len(input_files)+1,bins))

for i_f,file in enumerate(input_files):
    N_frames = 0
    with gsd.hoomd.open(file, 'r') as f:
        N_frames = f.__len__()

    frames = np.arange(N_frames)
    frames = frames[start_frame:end_frame]
    frame_data = Analyzer(file,0)
    N_shapes = frame_data.getNshapes()
    Lx = frame_data.getBoxSize()

    all_central_pos = np.zeros((N_frames,N_shapes,3))

    rdf = freud.density.RDF(bins, r_max)
    for ii,target_frame in enumerate(frames):
        frame_data = Analyzer(file,target_frame)
        pos_central_frame = frame_data.getCentralPos()
        all_central_pos[ii,:,:] = pos_central_frame
        box = np.array([box_size,box_size,box_size])
        rdf.compute(system=(box, pos_central_frame), reset=False)

    rdf_all[i_f,:] = rdf.rdf

rdf_all[-1,:] = rdf.bin_centers
outname = input_files[0][:-4] + '_rdf.txt'
np.savetxt(outname,rdf_all,fmt="%.5f",header="# last line is bin centers")

# plt.figure(1)
# plt.plot(rdf_all[-1,:],np.average(rdf_all[:-1],axis=0))
# plt.show()

# outname = input_file[:-4] + '_endata.txt'
# np.savetxt(outname,energy_data,fmt="%.5f",header="# e_tot pot e_kin trans_kin rot_kin trans_temp")


















exit()

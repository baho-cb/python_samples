from sys import argv
import sys
import os
sys.path.insert(0,"/home/baho/Desktop/scripts")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import argparse
import numpy as np
import gsd
import gsd.hoomd
from BreakUp import ClusterUtils
import re


"""
July 11
Dump the gyration tensor (diagonalized) at each frame through mpcd run to see whether
it is run sufficiently long or not
"""

parser = argparse.ArgumentParser(description="Cluster analysis of a gsd file during shearing")
non_opt = parser.add_argument_group("mandatory arguments")

non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_file",
required=True, help=".gsd files " )

non_opt.add_argument('--cutoff', metavar="<float>", type=float, dest="cutoff",
required=True, help=" cutoff for DBSCAN ", default = -1 )

non_opt.add_argument('--min_samples', metavar="<int>", type=int, dest="min_samples",
required=True, help=" min_samples for DBSCAN ..\
lower than this value wont count as a cluster but will be noise ",default=2)


args = parser.parse_args()
input_file = args.input_file
cutoff = args.cutoff
min_samples = args.min_samples

print("--- DBSCAN with ---")
print("Cutoff : ", cutoff)
print("Min_samples : ", min_samples)
print("File : ", input_file)
print("----------------------------")

in_file=gsd.fl.GSDFile(name=input_file, mode='rb',application="hoomd",schema="hoomd", schema_version=[1,0])
N_frames = in_file.nframes

dat = []

for i in range(N_frames):
    cluster_utils = ClusterUtils(input_file,i)
    ls = cluster_utils.dbscan(cutoff,min_samples)
    gyros = cluster_utils.get_gyros()
    dat.append(gyros)


dat = np.array(dat)
out_name = input_file[:-4] + ".gyros"
np.savetxt(out_name,dat,fmt="%.3f %.3f %.3f",header="# measure_gyration_tensor.py")



exit()

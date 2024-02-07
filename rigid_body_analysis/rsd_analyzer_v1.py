import numpy as np
import gsd.hoomd
import os.path
import sys
from AnalyzerUtils import Analyzer,wrap_pbc
import string
import argparse
import matplotlib.pyplot as plt
"""
August 30 - 2023

Calculate mean rotations to compare NN_simulator vs hoomd (angular equivalent of MSD)
Uses quaternion distance as a measure of rotation amount

https://math.stackexchange.com/questions/90081/quaternion-distance?noredirect=1&lq=1

python3 rsd_analyzer_v1.py -i simulation_output.gsd
"""

parser = argparse.ArgumentParser(description="rsd analysis")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_file",
required=True, help=".gsd files " )


args = parser.parse_args()
input_file = args.input_file

in_file=gsd.fl.GSDFile(name=input_file, mode='r',application="hoomd",schema="hoomd", schema_version=[1,0])
N_frames = in_file.nframes
frames = np.arange(N_frames)

frame_data = Analyzer(input_file,0)
N_shapes = frame_data.getNshapes()
Lx = frame_data.getBoxSize()

all_quats = np.zeros((N_frames,N_shapes,4))

for i_f,target_frame in enumerate(frames):
    frame_data = Analyzer(input_file,target_frame)
    quat_frame = frame_data.getOrientations()
    all_quats[i_f,:,:] = quat_frame


rsd = []
for i in range(1,N_frames-2):
    mt = all_quats[:-i,:,:]*all_quats[i:,:,:]
    mts = np.sum(mt,axis=2)
    qdist_i = 1.0 - mts*mts
    qdist_i_avg = np.average(qdist_i)
    rsd.append(qdist_i_avg)


rsd = np.array(rsd)
outname = input_file[:-4] + '_rsd.txt'
np.savetxt(outname, rsd, fmt='%.5f')


print("DONE")
exit()












exit()

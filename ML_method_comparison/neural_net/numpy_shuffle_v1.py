import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import numpy as np
import re
import time
import torch

"""
April 10

Takes in the merged numpy arrays (right now there is 4 of them)

-configs
-energies
-forces
-torks

"""

parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_datas",nargs="+", required=True, help="-i d1_configs_reduced_filtered.npy d2_configs_reduced_filtered.npy" )
non_opt.add_argument('--savename', metavar="<dat>", type=str, dest="savename", required=True )


args = parser.parse_args()
input_datas = args.input_datas
savename = args.savename


""" testing numpy.random.shuffle() """
# N = 13
# all_indexes = np.arange(N)
# N_train = np.int(N*0.70)
# # print(all_indexes)
# np.random.shuffle(all_indexes)
# # print(all_indexes)
# train_index = all_indexes[:N_train]
# test_index = all_indexes[N_train:]
#
# print(test_index)
# print(train_index)
#
# exit()

print(input_datas)

config_data = np.load(input_datas[0])

N = len(config_data)
all_indexes = np.arange(N)
N_train = int(N*0.70)
np.random.shuffle(all_indexes)
train_index = all_indexes[:N_train]
test_index = all_indexes[N_train:]

train_config = config_data[train_index]
test_config = config_data[test_index]

np.save('x_train.npy',train_config)
np.save('x_test.npy',test_config)
del(train_config,test_config,config_data)


energy_data = np.load(input_datas[1])

if(N!=len(energy_data)):
    print("Error 12")
    exit()

train_energy = energy_data[train_index]
test_energy = energy_data[test_index]

np.save('yen_train.npy',train_energy)
np.save('yen_test.npy',test_energy)
del(energy_data,train_energy,test_energy)



# force_data = np.load(input_datas[2])
#
# if(N!=len(force_data)):
#     print("Error 123")
#     exit()
#
# train_force = force_data[train_index]
# test_force = force_data[test_index]
#
# np.save('yf_train.npy',train_force)
# np.save('yf_test.npy',test_force)
# del(force_data,train_force,test_force)
#
#
# tork_data = np.load(input_datas[3])
#
# if(N!=len(tork_data)):
#     print("Error 133")
#     exit()
#
# train_tork = tork_data[train_index]
# test_tork = tork_data[test_index]
#
# np.save('yt_train.npy',train_tork)
# np.save('yt_test.npy',test_tork)
# del(tork_data,train_tork,test_tork)





exit()

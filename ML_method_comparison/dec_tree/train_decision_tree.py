import torch
import argparse
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import tree
import sys,os
import pickle
import time


np.set_printoptions(suppress=True)
"""
Jan 11

Trains the tree with training data and saves it for the test

There is no good method of saving a trained decision tree
-> So take the training (x,y), train the tree
-> Take the test (x,y), apply trained tree
-> Quantify the error (are the mismatches important?)
-> Save the finite portion of the test data (x,y) for NN test

First predict if the potential is Zero - Finite - Overlapping
                            Class  0        1         2

You can decide according to probabilities to stay on the safe side
(ie. if probability that finite class is > 0.1 for example take it to finite)
"""

parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', metavar="<dat>", type=str, dest="input_datas",nargs='+', required=True, help="-i x y training" )
non_opt.add_argument('--depth', metavar="<int>", type=int, dest="tree_depth", required=True, help="-i x y training" )
non_opt.add_argument('--N_rate', metavar="<float>", type=float, dest="N_rate", required=True, help="run id" )
non_opt.add_argument('--test', metavar="<dat>", type=str, dest="test_datas", nargs='+', required=True, help="-i x.pt y.pt" )

args = parser.parse_args()
input_datas = args.input_datas
tree_depth = args.tree_depth
N_rate = args.N_rate
test_datas = args.test_datas

x_np = np.load(input_datas[0])
y_np = np.load(input_datas[1])
y_class = np.ones_like(y_np,dtype=np.int32)
y_class[np.abs(y_np)<0.05] = 0
N_TOTAL = len(y_np)
NNN = int(N_TOTAL*N_rate)
x_np = np.copy(x_np[:NNN])
y_train = np.copy(y_class[:NNN])

"""Train"""
t0 = time.time()
clf = tree.DecisionTreeClassifier(max_depth=tree_depth)
clf = clf.fit(x_np, y_train)
tf = time.time() - t0
tree_name = 'dec_Nr%.3f_D%d.tree' %(N_rate,tree_depth)
with open(tree_name, "wb") as f:
    pickle.dump(clf, f)

"""Test"""
x_test = np.load(test_datas[0])
y_test = np.load(test_datas[1])
y_test_class = np.ones_like(y_test,dtype=np.int32)
y_test_class[np.abs(y_test)<0.05] = 0
t0_inf = time.time()
y_pred = clf.predict(x_test)
tf_inf = time.time() - t0_inf

err = np.abs(y_pred - y_test_class)
err_rate = len(err[err>0.01])/len(err)
print("%d %.3f %.5f %.3f %.3f"%(tree_depth,N_rate,err_rate,tf,tf_inf))

# tree_file = open('decision.tree', 'rb')
# clf = pickle.load(tree_file)
#
# y_pred = clf.predict(x)
# print(y_pred)
#
# # del(x)
# # del(y)
# # del(y_class)
# #
# # xt = torch.load(test_datas[0])
# # xt = xt.detach().numpy()
# #
# # yt = torch.load(test_datas[1])
# # yt = yt.detach().numpy()
# #
# # yt_class = np.ones(len(yt),dtype=np.int32)
# # yt_class[yt>12.0] = 2
# # yt_class[yt<0.05] = 0
# #
# # y_pred = clf.predict(xt)
#
# """error analysis"""
# """
# If the prediction result is class 1, we don't care it will go to the NN anyway
# We are more interested in predictions 0 and 2 because these may possibly cause
# trouble
# """
#
# diff = np.abs(y_pred-yt_class)
# wrong_class = diff[diff>0.05]
# N_wrong = len(wrong_class)
# accuracy = 1 - N_wrong/len(y_pred)
# wrong_index = np.where(diff>0.05)
# wrong_index = wrong_index[0]
#
# true_value_wrong = yt[wrong_index]
# true_class_wrong = yt_class[wrong_index]
# prediction_wrong = y_pred[wrong_index]
#
# error_1to0 = []
# error_1to2 = []
# error_0to2 = []
# error_2to0 = []
#
# for i,true in enumerate(true_value_wrong):
#     if(true_class_wrong[i]==1 and prediction_wrong[i]==0):
#         error_1to0.append(true)
#     if(true_class_wrong[i]==1 and prediction_wrong[i]==2):
#         error_1to2.append(true)
#     if(true_class_wrong[i]==0 and prediction_wrong[i]==2):
#         error_0to2.append(true)
#     if(true_class_wrong[i]==2 and prediction_wrong[i]==0):
#         error_2to0.append(true)
#
# error_1to0 = np.array(error_1to0)
# error_1to0 = np.sort(error_1to0)
#
# error_1to2 = np.array(error_1to2)
# error_1to2 = np.sort(error_1to2)
#
# error_0to2 = np.array(error_0to2)
# error_0to2 = np.sort(error_0to2)
#
# error_2to0 = np.array(error_2to0)
# error_2to0 = np.sort(error_2to0)
#
# print("Tree Depth : %d"%tree_depth)
# print("Decision Tree Accuracy : %.5f" %accuracy)
#
# print("- - - - Specific Error Rates - - - -")
# print("Finite values predicted as Zero : %f " %(len(error_1to0)/len(y_pred)) )
# print("Finite values predicted as Overlap : %f " %(len(error_1to2)/len(y_pred)))
# print("Zero values predicted as Overlap : %f " %(len(error_0to2)/len(y_pred)))
# print("Overlap values predicted as Zero : %f " %(len(error_2to0)/len(y_pred)))
#
# print("- - - - Max Errors - - - - ")
# print("Largest Finite values predicted as zero", error_1to0[-5:])
# print("Smallest Finite values predicted as Overlap", error_1to2[:5])
#
# print("- - - - Average Errors - - - - ")
# print("Average of Finite values predicted as zero %.5f" %np.mean(error_1to0))
# print("Average of Finite values predicted as Overlap %.5f" %np.mean(error_1to2))
#
# print(" - - - - - - - - ")
# finite_rate = len(y_pred[y_pred==1])/len(y_pred)
# print("Rate of Finite Values %.5f "%finite_rate)
# print('Saving the finite test data')
#
# """
# Save the finite part of the test data after adjusting tree depth
# """
# x_test_finite = xt[y_pred==1]
# y_test_finite = yt[y_pred==1]
#
# x_data = torch.from_numpy(x_test_finite)
# y_data = torch.from_numpy(y_test_finite)
#
# x_out = test_datas[0][:-3] + '_12interacting.pt'
# y_out = test_datas[1][:-3] + '_12interacting.pt'
#
# torch.save(x_data, x_out)
# torch.save(y_data, y_out)
#
#
#
#
#




exit()

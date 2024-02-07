from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
import argparse
import pickle

np.set_printoptions(precision=3, threshold=None, suppress=True)
# torch.set_printoptions(precision=6)

parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_datas", nargs='+', required=True, help="-i x.pt y.pt" )
non_opt.add_argument('--test', metavar="<dat>", type=str, dest="test_datas", nargs='+', required=True, help="-i x.pt y.pt" )
non_opt.add_argument('--N_rate', metavar="<float>", type=float, dest="N_rate", required=True, help="run id" )
non_opt.add_argument('--depth', metavar="<int>", type=int, dest="depth", required=True, help="run id" )
non_opt.add_argument('--n_estimators', metavar="<int>", type=int, dest="n_estimators", required=True, help="run id" )
args = parser.parse_args()
input_datas = args.input_datas
depth = args.depth
N_rate = args.N_rate
test_datas = args.test_datas
n_estimators = args.n_estimators

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
clf = RandomForestClassifier(max_depth=depth, random_state=0,n_estimators=n_estimators)
clf.fit(x_np, y_train)
tf = time.time() - t0
forest_name = 'rf_Nr%.3f_D%d_Nest%d.forest' %(N_rate,depth,n_estimators)
with open(forest_name, "wb") as f:
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
print("%d %d %.3f %.5f %.3f %.3f"%(depth,n_estimators,N_rate,err_rate,tf,tf_inf))






















exit()

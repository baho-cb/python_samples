#!/bin/bash



Ds=(5 10 15)
N_estim=(10 25 50)


for nest in "${N_estim[@]}"
do
  for D in "${Ds[@]}"
  do
    python3 random_forest_v1.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth $D --N_rate 0.05 --n_estimators $nest
    python3 random_forest_v1.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth $D --N_rate 0.1 --n_estimators $nest
  done
done








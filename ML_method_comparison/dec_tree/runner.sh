#!/bin/bash

python3 train_decision_tree.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth 5 --N_rate 0.05
python3 train_decision_tree.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth 5 --N_rate 0.1
python3 train_decision_tree.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth 5 --N_rate 0.2
python3 train_decision_tree.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth 10 --N_rate 0.05
python3 train_decision_tree.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth 10 --N_rate 0.1
python3 train_decision_tree.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth 10 --N_rate 0.2
python3 train_decision_tree.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth 15 --N_rate 0.05
python3 train_decision_tree.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth 15 --N_rate 0.1
python3 train_decision_tree.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth 15 --N_rate 0.2
python3 train_decision_tree.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth 20 --N_rate 0.05
python3 train_decision_tree.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth 20 --N_rate 0.1
python3 train_decision_tree.py -i ../x_train.npy ../yen_train.npy --test ../x_test.npy ../yen_test.npy --depth 20 --N_rate 0.2

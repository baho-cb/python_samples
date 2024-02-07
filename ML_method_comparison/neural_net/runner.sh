#!/bin/bash

Ws=(30 50 70)
Ds=(5 7 9)

for W in "${Ws[@]}"
do
  for D in "${Ds[@]}"
  do
    python3 -u trainer_torfor_v3.py -i x_train.npy yen_train.npy --stop 201 --lr 0.001 --batch_size 1024 --width $W --depth $D --gpu 1 --minutes 20 --N_rate 0.05
  done
done

for W in "${Ws[@]}"
do
  for D in "${Ds[@]}"
  do
    python3 -u trainer_torfor_v3.py -i x_train.npy yen_train.npy --stop 201 --lr 0.001 --batch_size 1024 --width $W --depth $D --gpu 1 --minutes 50 --N_rate 0.1
  done
done

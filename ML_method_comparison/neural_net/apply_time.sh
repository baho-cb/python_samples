#!/bin/bash
modelnames=`ls ./mmodels/model*.pth`
for modelname in $modelnames
do
	python3 apply_time.py -i x_test.npy --model ${modelname:2}
done

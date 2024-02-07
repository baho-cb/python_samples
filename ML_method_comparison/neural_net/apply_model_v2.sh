#!/bin/bash
modelnames=`ls ./models/model*.pth`
for modelname in $modelnames
do
	python3 apply_model_v2.py -i x_test.npy --model ${modelname:2}
done

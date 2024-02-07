#!/bin/bash
predictions=`ls ./raw_predictions/*.pt`
for prediction in $predictions
do
	python3 -u prediction_analysis.py --y_true ./yen_test.npy --y_pred ${prediction:2} 
done

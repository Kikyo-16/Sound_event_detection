#Code Contributor: Rohan Badlani, Email: rohan.badlani@gmail.com
import os
import sys
from src.evaluation.Models import *

def evaluateMetrics(groundtruth_filepath, predicted_filepath):
	groundTruthDS = FileFormat(groundtruth_filepath)
	predictedDS = FileFormat(predicted_filepath)
	
	#print(predictedDS.labelsDict)
	output = groundTruthDS.computeMetricsString(predictedDS)
	F1=output.split('F1 Score =')[-1].split('\n')[0]
	pre=output.split('Precision =')[-1].split('\n')[0]
	recall=output.split('Recall =')[-1].split('\n')[0]
	return float(F1)*0.01,float(pre)*0.01,float(recall)*0.01
if __name__ == "__main__":
	evaluateMetrics(sys.argv[1], sys.argv[2])

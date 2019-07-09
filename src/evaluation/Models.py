#Code Contributor: Rohan Badlani, Email: rohan.badlani@gmail.com
import os
import sys

class FileFormat(object):
	def __init__(self, filepath):
		self.count = 0 
		self.filepath = filepath
		self.labelDict = self.readLabels()

	def readLabels(self):
		try:
			#Filename will act as key and labels list will be the value 
			self.labelsDict = {}
			with open(self.filepath) as filename:
				for line in filename:
					lineArr = line.split("\t")
					#audioFile must be present, make it hard
					if len(lineArr) == 4:
						if float(lineArr[2].strip()) == 0.0:
							if(lineArr[0].split(".wav")[0].split(".flac")[0].strip() not in self.labelsDict.keys()): 
								self.count = self.count + 1
							continue
						else:
							if(lineArr[0].split(".wav")[0].split(".flac")[0].strip() not in self.labelsDict.keys()): 
								self.count = self.count + 1
							audioFile = lineArr[0].split(".wav")[0].split(".flac")[0].strip()
					else:
						if(lineArr[0].split(".wav")[0].split(".flac")[0].strip() not in self.labelsDict.keys()): 
							self.count = self.count + 1
						audioFile = lineArr[0].split(".wav")[0].split(".flac")[0].strip()

					try:
						startTime = lineArr[1].strip()
					except Exception as ex1:
						startTime = ""
					try:
						endTime = lineArr[2].strip()
					except Exception as ex2:
						endTime = ""
					try:
						label = lineArr[3].strip()
					except Exception as ex3:
						label = ""

					if audioFile not in self.labelsDict.keys():
						#does not exist
						if label is not "":
							self.labelsDict[audioFile] = [label]
						else:
							self.labelsDict[audioFile] = []
					else:
						#exists
						if label is not "":
							self.labelsDict[audioFile].append(label)

			filename.close()
			#Debug Print
			#for key in self.labelsDict.keys():
			#	print str(key) + ":" + str(self.labelsDict[key])

		except Exception as ex:
			print("Fileformat of the file " + str(self.filepath) + " is invalid.")
			raise ex

	def validatePredictedDS(self, predictedDS):
		#iterate over predicted list

		#check bothways
		for audioFile in predictedDS.labelsDict.keys():
			if(audioFile not in self.labelsDict.keys()):
				return False


		for audioFile in self.labelsDict.keys():
			if(audioFile not in predictedDS.labelsDict.keys()):
				return False
		
		#check complete. One-One mapping
		return True

	def computeMetrics(self, predictedDS, output_filepath):
		TP = 0
		FP = 0
		FN = 0

		classWiseMetrics = {}

		#iterate over predicted list
		for audioFile in predictedDS.labelsDict.keys():
			markerList = [0]*len(self.labelsDict[audioFile])
			for predicted_label in predictedDS.labelsDict[audioFile]:
				#for a predicted label
				
				#1. Check if it is present inside groundTruth, if yes push to TP, mark the existance of that groundtruth label
				index = 0
				for groundtruth_label in self.labelsDict[audioFile]:
					if(predicted_label == groundtruth_label):
						TP += 1
						markerList[index] = 1
						break
					index+=1

				if(index == len(self.labelsDict[audioFile])):
					#not found. Add as FP
					FP += 1
			
			#check markerList, add all FN
			for marker in markerList:
				if marker == 0:
					FN += 1

			for groundtruth_label in self.labelsDict[audioFile]:
				if groundtruth_label in predictedDS.labelsDict[audioFile]:
					#the class was predicted correctly
					if groundtruth_label in classWiseMetrics.keys():
						classWiseMetrics[groundtruth_label][0] += 1
					else:
						#Format: TP, FP, FN
						classWiseMetrics[groundtruth_label] = [1, 0, 0]
				else:
					#Not predicted --> FN
					if groundtruth_label in classWiseMetrics.keys():
						classWiseMetrics[groundtruth_label][2] += 1
					else:
						classWiseMetrics[groundtruth_label] = [0, 0, 1]

			for predicted_label in predictedDS.labelsDict[audioFile]:
				if predicted_label not in self.labelsDict[audioFile]:
					#Predicted but not in Groundtruth --> FP
					if predicted_label in classWiseMetrics.keys():
						classWiseMetrics[predicted_label][1] += 1
					else:
						classWiseMetrics[predicted_label] = [0, 1, 0]

		if(TP + FP != 0):
			Precision = float(TP) / float(TP + FP)
		else:
			Precision = 0.0
		if(TP + FN != 0):
			Recall = float(TP) / float(TP + FN)
		else:
			Recall = 0.0
		if(Precision + Recall != 0.0):
			F1 = 2 * Precision * Recall / float(Precision + Recall)
		else:
			F1 = 0.0

		
		with open(output_filepath, "w") as Metric_File:
			Metric_File.write("\n\nClassWise Metrics\n\n")	
		Metric_File.close()

		for classLabel in classWiseMetrics.keys():
			precision = 0.0
			recall = 0.0
			f1 = 0.0

			tp = classWiseMetrics[classLabel][0]
			fp = classWiseMetrics[classLabel][1]
			fn = classWiseMetrics[classLabel][2]
			if(tp + fp != 0):
				precision = float(tp) / float(tp + fp)
			if(tp + fn != 0):
				recall = float(tp) / float(tp + fn)
			if(precision + recall != 0.0):
				f1 = 2*precision*recall / float(precision + recall)

			with open(output_filepath, "a") as Metric_File:
				Metric_File.write("Class = " + str(classLabel) + ", Precision = " + str(precision) + ", Recall = " + str(recall) + ", F1 Score = " + str(f1) + "\n")
			Metric_File.close()
		#push to file
		with open(output_filepath, "a") as Metric_File:
			Metric_File.write("\n\nComplete Metrics\n\n")
			Metric_File.write("Precision = " + str(Precision*100.0) + "\n")
			Metric_File.write("Recall = " + str(Recall*100.0) + "\n")
			Metric_File.write("F1 Score = " + str(F1*100.0) + "\n")
			Metric_File.write("Number of Audio Files = " + str(self.count))
		Metric_File.close()

	
	def computeMetricsString(self, predictedDS):
		TP = 0
		FP = 0
		FN = 0

		classWiseMetrics = {}

		#iterate over predicted list
		for audioFile in predictedDS.labelsDict.keys():
			if(audioFile in self.labelsDict):
				markerList = [0]*len(self.labelsDict[audioFile])
				for predicted_label in predictedDS.labelsDict[audioFile]:
					#for a predicted label
					
					#1. Check if it is present inside groundTruth, if yes push to TP, mark the existance of that groundtruth label
					index = 0
					for groundtruth_label in self.labelsDict[audioFile]:
						if(predicted_label == groundtruth_label and markerList[index] != 1):
							TP += 1
							markerList[index] = 1
							break
						index+=1

					if(index == len(self.labelsDict[audioFile])):
						#not found. Add as FP
						FP += 1
			
				#check markerList, add all FN
				for marker in markerList:
					if marker == 0:
						FN += 1

				for groundtruth_label in self.labelsDict[audioFile]:
					if groundtruth_label in predictedDS.labelsDict[audioFile]:
						#the class was predicted correctly
						if groundtruth_label in classWiseMetrics.keys():
							classWiseMetrics[groundtruth_label][0] += 1
						else:
							#Format: TP, FP, FN
							classWiseMetrics[groundtruth_label] = [1, 0, 0]
					else:
						#Not predicted --> FN
						if groundtruth_label in classWiseMetrics.keys():
							classWiseMetrics[groundtruth_label][2] += 1
						else:
							classWiseMetrics[groundtruth_label] = [0, 0, 1]

				for predicted_label in predictedDS.labelsDict[audioFile]:
					if predicted_label not in self.labelsDict[audioFile]:
						#Predicted but not in Groundtruth --> FP
						if predicted_label in classWiseMetrics.keys():
							classWiseMetrics[predicted_label][1] += 1
						else:
							classWiseMetrics[predicted_label] = [0, 1, 0]
		if(TP + FP != 0):
			Precision = float(TP) / float(TP + FP)
		else:
			Precision = 0.0
		if(TP + FN != 0):
			Recall = float(TP) / float(TP + FN)
		else:
			Recall = 0.0
		if(Precision + Recall != 0.0):
			F1 = 2 * Precision * Recall / float(Precision + Recall)
		else:
			F1 = 0.0
		output = ""
		output += "\n\nClass-wise Metrics\n\n"

		classWisePrecision = 0.0
		classWiseRecall = 0.0
		classWiseF1 = 0.0
		classCount = 0
		for classLabel in classWiseMetrics.keys():
			classCount += 1

			precision = 0.0
			recall = 0.0
			f1 = 0.0

			tp = classWiseMetrics[classLabel][0]
			fp = classWiseMetrics[classLabel][1]
			fn = classWiseMetrics[classLabel][2]
			if(tp + fp != 0):
				precision = float(tp) / float(tp + fp)
				classWisePrecision += precision
			if(tp + fn != 0):
				recall = float(tp) / float(tp + fn)
				classWiseRecall += recall
			if(precision + recall != 0.0):
				f1 = 2*precision*recall / float(precision + recall)

			output += "\tClass = " + str(classLabel.split("\n")[0]) + ", Precision = " + str(precision) + ", Recall = " + str(recall) + ", F1 Score = " + str(f1) + "\n"
		classWisePrecision = classWisePrecision / classCount
		classWiseRecall = classWiseRecall / classCount
		if(classWisePrecision + classWiseRecall != 0.0):
			classWiseF1 = 2*classWisePrecision*classWiseRecall / float(classWisePrecision + classWiseRecall)

		output += "\n\n\tComplete Metrics (Macro Average or Class-Based)\n\n"
		output += "\tPrecision = " + str(classWisePrecision*100.0) + "\n"
		output += "\tRecall = " + str(classWiseRecall*100.0) + "\n"
		output += "\tF1 Score = " + str(classWiseF1*100.0) + "\n"
		output += "\tNumber of Audio Files = " + str(predictedDS.count) + "\n\n"
		
		output += "\n\n\tComplete Metrics (Micro Average or Instance-Based) - These metrics will be used for system evaluation.\n\n"
		output += "\tPrecision = " + str(Precision*100.0) + "\n"
		output += "\tRecall = " + str(Recall*100.0) + "\n"
		output += "\tF1 Score = " + str(F1*100.0) + "\n"
		output += "\tNumber of Audio Files = " + str(predictedDS.count) + "\n\n"
				
		return output




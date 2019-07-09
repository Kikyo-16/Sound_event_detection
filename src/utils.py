from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage
import dcase_util
import os
import numpy as np
import configparser
import scipy
from src.evaluation import sound_event_eval
from src.evaluation import scene_eval
import src.evaluation.TaskAEvaluate as taskAEvaluate
import copy
from scipy.ndimage.filters import median_filter
 
#duration of a single audio file (second)
DURATION=10.0
class utils(object):
	def __init__(self,conf_dir,
			exp_dir,
			label_lst):
		""""
		Tools to calculate performace.
		Args:
			conf_dir: string
				the path of configuration dir
			exp_dir: string
				the path of experimental dir
			label_lst: list
				the event list
		Attributes:
			conf_dir
			label_lst
			evaluation_path
			evaluation_ests
			evaluation_refs
			preds_path
			CLASS
			win_lens
			metric
			ave
			
		Interface:	
		
		"""
		self.conf_dir=conf_dir
		self.label_lst=label_lst
		self.init_utils_conf()

		self.evaluation_path=os.path.join(exp_dir,'evaluation')	
		self.init_dirs(self.evaluation_path)
		self.evaluation_ests=os.path.join(self.evaluation_path,'ests')
		self.init_dirs(self.evaluation_ests)
		self.evaluation_refs=os.path.join(self.evaluation_path,'refs')
		self.init_dirs(self.evaluation_refs)
		self.preds_path=os.path.join(self.evaluation_path,'preds.csv')

	def init_dirs(self,path):
		""""
		Create new dir.
		Args:
			path: string
				the path of the dir to create
		Return:

		"""
		if not os.path.exists(path):
			os.mkdir(path)

	def set_win_lens(self,win_lens):
		""""
		Set adaptive sizes of median windows.
		Args: list
			adaptive sizes of median windows
		Return

		"""
		if not len(self.win_lens) == self.CLASS:
			self.win_lens=win_lens

	def init_utils_conf(self):
		"""""
		Initialize most of attribute values from the configuration file.
		Args:
		Return:

		"""
		CLASS=len(self.label_lst)
		self.CLASS=CLASS
		conf_dir=self.conf_dir
		utils_cfg_path=os.path.join(conf_dir,'train.cfg')

		assert os.path.exists(utils_cfg_path)
		config=configparser.ConfigParser()
		config.read(utils_cfg_path)
		conf=config['validate']
		win_len=conf['win_len']
		if not win_len=='auto':
			self.win_lens=np.array([int(conf['win_len'])]*CLASS)
		else:
			self.win_lens=[]

		self.metric=conf['metric']
		self.ave=conf['ave']



	def get_vali_lst(self):
		""""
		Get current file list and groundtruths using for calculating 
		performance.
		Args:
		Return:
			lst: list
				the path list of files
			csv: list
				the groundtruth list of files

		"""
		lst=self.lst
		csv=self.csv
		return lst,csv

	def set_vali_csv(self,lst,csv):
		""""
		Set current file list and groundtruths for calculating performance.
		Args:
			lst: list
				the path list of files
			csv: list
				the groundtruth list of files
		Return:

		"""
		self.lst=lst
		self.csv=csv
			
	def init_csv(self,csvs,flag=True):
		""""
		Format groundtruths from a csv file to several single files.
		All the delimiters should be '\t'.
		Eg.
		original file:
			file_ori:
				A.wav	0.00	1.00	Cat
				A.wav	1.00	2.00	Dog	
				B.wav	0.00	1.00	Dog
		target files:
			file1: A.txt
				0.00    1.00    Cat
				1.00    2.00    Dog
			file2: B.txt
				0.00    1.00    Dog
		Args:
			csvs: list
				the groundtruth list to format
			flag: bool
				If flag is true, save result files into 
				evaluation_refs dir.
				Otherwise, save result files into
				evaluation_ests dir.
			
		"""
		if flag:
			root=self.evaluation_refs
		else:
			root=self.evaluation_ests
		#get formatted results
		result=self.format_lst(csvs)
		#save formatted results
		self.format_csv(result,root)



	def format_lst(self,csvs):
		""""
		Format the groundtruths.
		Eg.
		ori list:
			['A.wav   0.00    1.00    Cat',
			 'A.wav   1.00    2.00    Dog',
			 'B.wav   0.00    1.00    Dog']
		obj dict:
			{'A':[['0.00','1.00','Cat'],
			      ['1.00','2.00','Dog']],
			 'B':[['0.00','1.00','Dog']]}
		Args:
			csv: list
				the groundtruth list to format
		Return:
			result: dict
				formatted results
				
		
		"""
		tests=[t.rstrip().split('\t') for t in csvs]
		result={}
		cur=0
		for i,t in enumerate(tests):
			f=str.replace(t[0],'.wav','')
			if f not in result:
				result[f]=[]
			if len(t)>1:
				result[f]+=[[t[1],t[2],t[3]]]
			else:
				cur+=1
		return result
	
	def format_csv(self,tests,root):
		""""
		Save formatted results to several files.
		Eg.
		ori dict:
			{'A':[['0.00','1.00','Cat'],
                              ['1.00','2.00','Dog']],
			 'B':[['0.00','1.00','Dog']]}
		obj files:
			file1: A.txt
				0.00    1.00    Cat
				1.00    2.00    Dog
			file2: B.txt
				0.00    1.00    Dog
		Args:
			tests: list
				formatted results
			root: string
				the path of dir to save files
		Return:
		
		"""
		for t in tests:
			#get the path of a single file
			fname=os.path.join(root,t)
			with open(fname+'.txt','w') as f:
				result=[]
				for k in tests[t]:
					if len(k)>1:
						result+=['%s\t%s\t%s'%(k[0],k[1],k[2])]
				result='\n'.join(result)
				f.writelines(result)
			



	def get_f1(self,preds,labels,mode='at'):
		""""
		Calculate perfomance.
		Args:
			preds: numpy.array
				clip-level predicton (posibilities)
			labels: numpy.array
				weakly labels (or frame-level predicton)
			mode: string in ['at','sed']
				get audio tagging perfomance in mode 'at' and 
				get sound event perfomance in mode 'sed'
		Return:
			if mode=='at':
				F1: numpy.array
					clip-level F1 score
				precision: numpy.array
					clip-level precision
				recall: numpy.array
					clip-level recall
			if mode=='sed':
				segment_based_metrics: sed_eval.sound_event.SegmentBasedMetrics
					segment based result
				event_based_metrics: sed_eval.sound_event.event_based_metrics
					event based result

		"""
		#get current file list
		lst,_=self.get_vali_lst()
		preds=preds[:len(lst)]
		#using threshold of 0.5 to get clip-level decision
		preds[preds>=0.5]=1
		preds[preds<0.5]=0

		evaluation_path=self.evaluation_path
		
		#get audio tagging performance
		if mode=='at':
			labels=labels[:len(lst)]	
			ave=self.ave
			CLASS=labels.shape[-1]
			TP=(labels+preds==2).sum(axis=0)
			FP=(labels-preds==-1).sum(axis=0)
			FN=(labels-preds==1).sum(axis=0)
			if ave=='class_wise_F1':
				TFP=TP+FP
				TFP[TFP==0]=1
				precision=TP/TFP
				TFN=TP+FN
				TFN[TFN==0]=1
				recall=TP/TFN
				pr=precision + recall
				pr[pr==0]=1
			elif ave=='overall_F1':
				TP=np.sum(TP)
				FP=np.sum(FP)
				FN=np.sum(FN)

				TFP=TP+FP
				if TFP==0:
					TFP=1
				precision=TP/TFP
				TFN=TP+FN
				if TFN==0:
					TFN=1
				recall=TP/TFN
				pr=precision + recall
				if pr==0:
					pr=1

			F1=2*precision*recall/pr

			if ave=='class_wise_F1':
				class_wise_f1=F1
				class_wise_pre=precision
				class_wise_recall=recall
				F1=np.mean(F1)
				precision=np.mean(precision)
				recall=np.mean(recall)

			return F1,precision,recall,class_wise_f1,class_wise_pre,class_wise_recall

		#get event detection performance
		elif mode=='sed':
			segment_based_metrics,event_based_metrics=self.get_sed_result(preds,labels)
			return segment_based_metrics,event_based_metrics
		assert False
			
		

	def get_predict_csv(self,results):
		""""
		Format all the results into a file.
		Eg.
                ori dict:
			{'A':[['0.00','1.00','Cat'],
                              ['1.00','2.00','Dog']],
			 'B':[['0.00','1.00','Dog']],
			 'C':[]}
		obj file content:
			A.wav   0.00    1.00    Cat
			A.wav   1.00    2.00    Dog
			B.wav   0.00    1.00    Dog
			C.wav			
		
		Args:
			results: dict
				original dict to format
		Return:
			outs: list
				content of the file to save
				
		"""
		outs=[]
		for re in results:
			flag=True
			for line in results[re]:
				outs+=['%s.wav\t%s\t%s\t%s'%(
					re,line[0],line[1],line[2])]
				flag=False
			if flag:
				outs+=['%s.wav\t\t\t'%re]
		with open(self.preds_path,'w') as f:
			f.writelines('\n'.join(outs))
		return outs
	
	def get_sed_result(self,preds,frame_preds):
		""""
		Calculate event detection performance.
		Args:
			preds: numpy.array
				clip-level decision
			frame_preds: numpy.array
				frame-level prediction
		Return:
			segment_based_metrics: sed_eval.sound_event.SegmentBasedMetrics
				segment based result
			event_based_metrics: sed_eval.sound_event.EventBasedMetrics
				event based result
			
		"""

		#get current file list
		lst,csv=self.get_vali_lst()
		label_lst=self.label_lst
		win_lens=self.win_lens
		CLASS=self.CLASS
		#get the number of frames of the frame-level predicion
		top_LEN=frame_preds.shape[1]
		#duration (second) per frame
		hop_len=DURATION/top_LEN

		frame_preds=frame_preds[:len(lst)]

		decision_encoder=dcase_util.data.DecisionEncoder(
			label_list=label_lst)

		shows=[]
		result={}
		file_lst=[]

		for i in range(len(lst)):
			pred=preds[i]
			frame_pred=frame_preds[i]
			for j in range(CLASS):
				#If there is not any event for class j
				if pred[j]==0:
					frame_pred[:,j]*=0
				else:
					#using median_filter on prediction for the first post-processing
					frame_pred[:,j]=median_filter(
                                            frame_pred[:,j],(win_lens[j]))	
			#making frame-level decision
			frame_decisions=dcase_util.data.ProbabilityEncoder()\
				.binarization(
					probabilities=frame_pred,
					binarization_type='global_threshold',
					time_axis=0)

			# using median_filter on decision for the second post-processing
			for j in range(CLASS):
				frame_decisions[:,j]=median_filter(
					frame_decisions[:,j], (win_lens[j]))
			
			#generate reference-estimated pairs
			if lst[i] not in result:
				result[lst[i]]=[]
				file_lst+=[{'reference_file':'refs/%s.txt'%lst[i],
					'estimated_file':'ests/%s.txt'%lst[i]}]

			#encode discrete decisions to continuous decisions 
			for j in range(CLASS):
				estimated_events=decision_encoder\
						.find_contiguous_regions(
					activity_array=frame_decisions[:,j])
				
				for [onset, offset] in estimated_events:
					result[lst[i]]+=[[str(onset*hop_len),
							str(offset*hop_len),
							label_lst[j]]]
		#save continuous decisions to a file
		self.get_predict_csv(result)
		#save continuous decisions to multiple files for evaluation
		self.format_csv(result,self.evaluation_ests)

		#get performance using dcase_util
		segment_based_metrics,event_based_metrics=sound_event_eval.main(
			self.evaluation_path,file_lst)

		return segment_based_metrics,event_based_metrics

		

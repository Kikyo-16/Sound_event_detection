import copy
import configparser
from keras import backend as K
import keras
from keras import objectives
import tensorflow as tf
from keras.models import load_model, Model
import os
import numpy as np
import random
import shutil
import sys
from src import data_loader as data
from src import model as md
from src import utils
from src import metricCallback
from src.Logger import LOG

class trainer(object):
	def __init__(self, task_name, model_name, from_exp):
		""""
		Help configure data flow loading, training, and testing processes
		and model building.
		Args:
			task_name: string
				the name of the task	
			model_name: string
				the name of the model
			from_exp: bool
				whether to load the model from the file
		Attributes:
			task_name
			model_name
			resume_training
			conf_dir
			data_loader
			model_struct
			utils
			exp_dir
			result_dir
			exp_conf_dir
			best_model_path

		Interface:
			init_train_conf: Initialize most of attribute values from the configuration file.
			init_data: Initialize a src.model object.
			init_model: Initialize a src.model object.
			prepare_exp: Prepare experimental dirs and model path.
			init_utils: Initialize a src.utils object.
			prepare_attributes: Fill in some attribute values in utils and model_struct.
			get_metricCallback: Initialize a src.metricCallback object for training.
			prepare_exp: Prepare experimental dirs and model path.	
			
			train: Implement training.
			test: Get prediction on the specified dataset.
			save_at_result: Predict and save audio tagging performance both on validation set and test set.
			save_at:
			save_sed_result: Predict and save event detection performance both on validation set and test set.
			save_sed:
			save_str: Save a list of strings into a file.
				
				
		"""
		self.task_name = task_name
		self.model_name = model_name
		self.resume_training = from_exp

		#Determine whether to load the configuration file from the experimental dir.
		if from_exp:
			self.conf_dir = os.path.join('exp', task_name, model_name, 'conf')
		else:
			self.conf_dir = os.path.join(task_name, 'conf')

		self.init_train_conf()
		#Set data_loader
		self.init_data()
		#Set model_struct
		self.init_model()
		#Prepare experimental dirs
		self.prepare_exp()
		#[prepare_exp] must be performed before [init_utils]
		self.init_utils()
		#Fill in some attribute values in utils and model_struct
		self.prepare_attributes()


	def init_train_conf(self):
		"""""
		Initialize most of attribute values from the configuration file.
		Args:
		Return:

		"""	
		conf_dir = self.conf_dir
		train_cfg_path = os.path.join(conf_dir, 'train.cfg')
		assert os.path.exists(train_cfg_path)
		config = configparser.ConfigParser()
		config.read(train_cfg_path)	
	
		assert 'trainer' in config.sections()
		train_conf = config['trainer']
		self.epochs = int(train_conf['epochs'])
		assert 'validate' in config.sections()
		vali_conf = config['validate']

	def init_model(self):
		""""
		Initialize a src.model object.
		Args:
		Return:

		"""
		conf_dir = self.conf_dir
		model_name = self.model_name
		task_name = self.task_name
		data_loader = self.data_loader
		self.model_struct = md.attend_cnn(conf_dir, model_name, task_name, 
			data_loader.LEN, data_loader.DIM, data_loader.CLASS)

	def init_data(self):
		""""
		Initialize a src.data_loader object.
		Args:
		Return:

		"""
		conf_dir = self.conf_dir
		self.data_loader = data.data_loader(conf_dir)

	def init_utils(self):
		""""
		Initialize a src.utils object.
		Args:
		Return:
		
                """
		conf_dir = self.conf_dir
		data_loader = self.data_loader
		exp_dir = self.exp_dir
		self.utils = utils.utils(conf_dir, exp_dir, data_loader.events)
		lst, csv = data_loader.get_test()
		self.utils.init_csv(csv)
		lst, csv = data_loader.get_vali()
		self.utils.init_csv(csv)
		self.utils.set_vali_csv(lst, csv)

	def prepare_attributes(self):
		""""
		Fill in some attribute values in utils and model_struct.
		Args:
		Return:

		"""
		data_loader = self.data_loader
		model_struct = self.model_struct
		utils_obj = self.utils

		dfs = data_loader.count_disentangle()
		model_struct.set_DFs(dfs)

		win_lens = data_loader.count_win_len_per_class(model_struct.top_len)
		utils_obj.set_win_lens(win_lens)	

	def get_metricCallback(self, train_mode):
		""""
		Initialize a src.metricCallback object for training.
		Args:
			train_mode: string in ['semi','supervised']
				semi-supervised learning or weakly-supervised learning
		Return:
			callbacks: src.metricCallback
				the target src.metricCallback object
		"""
		data_loader = self.data_loader
		callback = metricCallback.metricCallback(self.conf_dir, train_mode)
		callback.set_extra_attributes(self.utils, 
						self.best_model_path, 
						data_loader.batch_size, 
						data_loader.CLASS)
		return callback

	def prepare_exp(self):
		""""
		Prepare experimental dirs and model path.
		Args:
		Return:
		
		"""
		model_name = self.model_name
		task_name = self.task_name
		resume_training = self.resume_training
		conf_dir = self.conf_dir

		#If the experimental dir doesn't exist, then create
		if not os.path.exists('exp'):
			os.mkdir('exp')

		#If task dir in the experimental dir doesn't exist, then create
		root_dir = os.path.join('exp', task_name)
		if not os.path.exists(root_dir):
			os.mkdir(root_dir)

		#prepare several dirs
		exp_dir = os.path.join(root_dir, model_name)
		model_dir = os.path.join(exp_dir, 'model')
		result_dir = os.path.join(exp_dir, 'result')
		exp_conf_dir = os.path.join(exp_dir, 'conf')

		self.exp_dir = exp_dir
		self.result_dir = result_dir
		self.exp_conf_dir = exp_conf_dir

		self.best_model_path = os.path.join(model_dir, 'best_model_w.h5')


		#If retrain, then renew all the dirs
		#Or check whether all the required dirs and model path exist
		if not resume_training:
			if os.path.exists(exp_dir):
				shutil.rmtree(exp_dir)
			os.mkdir(exp_dir)
			os.mkdir(model_dir)
			os.mkdir(result_dir)
			shutil.copytree(conf_dir, exp_conf_dir)
			
		else:
			assert os.path.exists(exp_dir)
			assert os.path.exists(exp_conf_dir)
			assert os.path.exists(model_dir)
			assert os.path.exists(self.best_model_path)
			if not os.path.exists(result_dir):
				os.mkdir(result_dir)



	def train(self, extra_model = None, train_mode = 'semi'):
		""""
		Implement training.
		Args:
			extra_model: Model
				the model structure to train
				(if None, take the default model structure)
			train_mode: string in ['semi','supervised']
				semi-supervised learning or weakly-supervised learning	
		Return:

		"""	

		resume_training = self.resume_training
		model_name = self.model_name
		data_loader = self.data_loader
		#total training epochs
		epochs = self.epochs

		#Callback
		callback = self.get_metricCallback(train_mode)

		if extra_model is not None:
			model = extra_model
		else:
			model = model_struct.get_model()

		#If resume training, load the model weights from the file
		if resume_training:
			model.load_weights(self.best_model_path, by_name = True)

		#Compile the model, actually it doesn't do anything.
		#The actual compilation takes place at the beginning of the training.
		model.compile(optimizer = 'Adam', loss = 'binary_crossentropy')

		#get data generator and the number of steps per epoch
		gt, steps_per_epoch = self.data_loader.generator_train()

		#get validation data
		vali_data = self.data_loader.generator_vali()

		#training using callback
		model.fit_generator(gt(), steps_per_epoch = steps_per_epoch, 
			epochs = epochs, shuffle = False, 
			validation_data = vali_data, callbacks = [callback])


	

	def test(self, data_set, mode, preds = {}):
		""""
		Get prediction on the specified dataset.
		Args:
			data_set: string in ['vali','test']
				prediction maken on the validation set with 'vali'
				and on the test set with 'test'
			mode: string in ['at','sed']
				'at' for clip-level prediction and 'sed' for both
				clip-level prediction and frame-level prediction
			preds: dict (eg. {'test':numpy.array, 'vali':numpy.array}
					or {})
				clip-level prediction (when mode is 'at') or
				frame-level prediction (when mode is 'sed')
				when data_set in dict is not None, nothing to do
				with predicting from model and take preds as
				prediction directly
		Return:
			mode == 'at':
				preds: numpy.array
					clip-level prediction
				label: numpy.array
					weakly-labeled data
			mode == 'sed':
				at_pred: numpy.array
					clip-level prediction
				sed_pred: numpy.array
					frame-level prediction

		"""
		data_loader = self.data_loader
		assert data_set == 'vali' or data_set == 'test'

		#get data from dataset
		if data_set == 'vali':
			data = data_loader.generator_vali()
		else:
			data = data_loader.generator_test()

		assert mode == 'at' or mode == 'sed'
		best_model_path = self.best_model_path

		#predict
		if mode == 'at':
			if not data_set in preds:
		
				model = self.model_struct.get_model(
					pre_model = best_model_path, 
					mode = mode)
				preds = model.predict(data[0], batch_size = data_loader.batch_size)
			else:
				preds = preds[data_set]
			label = data[1][:, :data_loader.CLASS]
			return preds, label
		else:
			if not data_set in preds:
				model = self.model_struct.get_model(
					pre_model = best_model_path, 
					mode = mode)
				preds = model.predict(data[0], batch_size = data_loader.batch_size)
			else:
				preds = preds[data_set]

			at_pred = preds[0]
			sed_pred = preds[1]
			return at_pred, sed_pred
		
	
	def save_at_result(self, at_preds = {}):
		""""
		Predict and save audio tagging performance both on validation set and test set.
		Args:
			at_preds: dict
				{'vali': numpy.array, 'test': numpy.array} or {}
				prediction (possibilities) on both set
				
		Return:
			preds_out: dict
				{'vali': numpy.array, 'test': numpy.array}
				prediction (possibilities) on both set
		"""
		preds_out = {}
		#get prediction (possibilities) on validation set and save results
		preds_out['vali'] = self.save_at('vali', at_preds, is_add = False)
		#get prediction (possibilities) on test set and save results
		preds_out['test'] = self.save_at('test', at_preds, is_add = True)
		return preds_out
		

	def save_at(self, mode = 'test', at_preds = {}, is_add = False):
		""""
		Args:
			mode: string in ['vali','test']
				the dataset to predict
			at_preds: dict
				If there is no prediction for the current data set 
				contained in the at_preds, the prediction will be 
				generated by the model.
				Otherwise the prediction in the at_preds is 
				considered as the prediction of the model.
				
			is_add: bool
				whether to open the result files by append
		Return:
			preds_ori: numpy.array
				prediction (possibilities)
		"""
		result_dir = self.result_dir
		model_name = self.model_name
		data_loader = self.data_loader
		f1_utils = self.utils
		result_path = os.path.join(result_dir, model_name + '_at.txt')
		detail_at_path = os.path.join(result_dir, model_name + '_detail_at.txt')
		#load the file list and the groundtruths
		if mode == 'vali':
			lst, csv = data_loader.get_vali()
		elif mode == 'test':
			lst, csv = data_loader.get_test()

		#prepare the file list and the groundtruths for counting scores
		f1_utils.set_vali_csv(lst, csv)

		#get clip-level prediction and weakly-labeled data
		preds, labels = self.test(mode, 'at', at_preds)
		preds_ori = copy.deepcopy(preds)
		#get F1 performance
		f1, precision, recall, cf1, cpre, crecall = f1_utils.get_f1(preds, labels, mode = 'at')	
		outs = []
		#result string to show and save
		outs += ['[ result audio tagging %s f1 : %f, precision : %f, recall : %f ]'
						%(mode, f1, precision, recall)]

		#show result
		for o in outs:
			LOG.info(o)

		data_loader = self.data_loader
		label_lst = data_loader.events
		details = []
		for i in range(len(label_lst)):
			line = '%s\tf1: %f\tpre: %f\trecall: %f'%(label_lst[i], 
							cf1[i], cpre[i], crecall[i])
			details += [line]

		#save result
		self.save_str(result_path, outs, is_add)
		self.save_str(detail_at_path, details, is_add)

		#return clip-level prediction (posibilities)
		return preds_ori


	def save_sed_result(self, sed_preds = {}):
		""""
                Predict and save event detection performance both on validation set
		and test set.
		Args:
			sed_preds: dict
				{'vali': numpy.array, 'test': numpy.array} or {}
				prediction (possibilities) on both set

		Return:
			preds_out: dict
				{'vali': numpy.array, 'test': numpy.array}
				prediction (possibilities) on both set
		"""
		preds_out = {}
		preds_out['vali'] = self.save_sed(mode = 'vali', 
					sed_preds = sed_preds, is_add = False)
		preds_out['test'] = self.save_sed(mode = 'test', 
					sed_preds = sed_preds, is_add = True)
		return preds_out

	def save_sed(self, mode = 'test', sed_preds = {}, is_add = False):
		""""
		Args:
			mode: string in ['vali','test']
				the dataset to predict
			at_preds: dict
				If there is no prediction for the current data set
				contained in the sed_preds, the prediction will be
				generated by the model.
				Otherwise the prediction in the at_preds is
				considered as the prediction of the model.

			is_add: bool
				whether to open the result files by append
		Return:
			preds_ori: numpy.array
				prediction (possibilities)
                """
		model_path = self.best_model_path
		result_dir = self.result_dir
		model_name = self.model_name

		data_loader = self.data_loader
		f1_utils = self.utils

		result_path = os.path.join(result_dir, model_name + '_sed.txt')
		detail_sed_path = os.path.join(result_dir, 
						model_name + '_detail_sed.txt')
		#path to save prediction (fomatted string)
		preds_csv_path = os.path.join(result_dir, 
						model_name + '_%s_preds.csv'%mode)

		#get clip-level prediction and frame-level prediction
		preds, frame_preds = self.test(mode, 'sed', sed_preds)
		ori_frame_preds = copy.deepcopy(frame_preds)

		outs = []

		#load the file list and the groundtruths
		if mode == 'vali':
			lst, csv = data_loader.get_vali()
		else:
			lst, csv = data_loader.get_test()

		#prepare the file list and the groundtruths for counting scores
		f1_utils.set_vali_csv(lst, csv)

		#get F1 performance (segment_based and event_based)
		segment_based_metrics, event_based_metrics = f1_utils.get_f1(
			preds, frame_preds, mode = 'sed')

		seg_event = [segment_based_metrics, event_based_metrics]
		seg_event_str = ['segment_based','event_based']

		
		for i, u in enumerate(seg_event):
			re = u.results_class_wise_average_metrics()
			f1 = re['f_measure']['f_measure']
			er = re['error_rate']['error_rate']
			pre = re['f_measure']['precision']
			recall = re['f_measure']['recall']
			dele = re['error_rate']['deletion_rate']
			ins = re['error_rate']['insertion_rate']
			outs += ['[ result sed %s %s macro f1 : %f, er : %f, pre : %f, recall : %f, deletion : %f, insertion : %f ]'%(mode, seg_event_str[i], f1, er, pre, recall, dele, ins)]

		#show result
		for o in outs:
			LOG.info(o)

		#save result
		self.save_str(result_path, outs, is_add)

		#save class-wise performaces into a file
		for u in seg_event:
			self.save_str(detail_sed_path, [u.__str__()], is_add)
			is_add = True

		#copy prediction csv file from evaluation dir to result dir
		shutil.copyfile(f1_utils.preds_path, preds_csv_path)

		preds = np.reshape(preds, [preds.shape[0], 1, preds.shape[1]])

		#return frame-level prediction (probilities)
		return ori_frame_preds * preds

	def save_str(self, path, content, is_add = False):
		""""
		Save a list of strings into a file.
		Args:
			path: string
				the path of the file to save
			content: list
				the list of strings to save
			is_add: bool
				whether to open the file by append
		Return:

		"""
		content += ['']
		if is_add:
			a = 'a'
		else:
			a = 'w' 
		with open(path, a) as f:
			f.writelines('\n'.join(content))
	

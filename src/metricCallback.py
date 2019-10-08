import configparser
from keras import backend as K
import keras
from keras import objectives
import tensorflow as tf
from keras.models import load_model, Model
import os
import numpy as np
from src import utils
from src.Logger import LOG

class metricCallback(keras.callbacks.Callback):
	def __init__(self, conf_dir, train_mode = 'semi'):
		""""
		MetricCallback for training.
		Args:
			conf_dir: string
				the path of configuration dir
			train_mode: string in ['semi','supervised']
				semi-supervised learning or weakly-supervised learning
		Attributes:
			conf_dir
			train_mode
			learning_rate
			decay_rate
			epoch_of_decay
			early_stop
			metric
			ave
			f1_utils
			best_model_path
			batch_size
			CLASS
			best_f1
			best_epoch
			wait
		Interface:
			set_extra_attributes: Set several required attributes.
			init_attributes: Set default values to some attributes.
			check_attributes: Check whether some required attributes have been set.
			init_train_conf: Initialize most of attribute values from the configuration file.
			get_at: Count audio tagging performance (F1).
			get_opt: Optimizer with specified learning rate.
			get_loss: Loss function for semi-supervised learning and weakly-supervised learning.
			on_train_begin
			on_epoch_end
			on_train_end

                """	
		self.train_mode = train_mode
		assert train_mode == 'semi' or train_mode == 'supervised'
		self.conf_dir = conf_dir
		self.init_train_conf()
		self.init_attributes()
		super(metricCallback, self).__init__()

		
	def set_extra_attributes(self, f1_utils, best_model_path, batch_size, CLASS):
		""""
		Set several required attributes.
		Args:
			f1_utils: src.utils
				a tool to calculate F-meansure
			best_model_path: string
				the path to save the best performance model 
			batch_size: integer
				the size of a batch
			CLASS: integer
				the number of event categories
		Return:
		"""
		self.f1_utils = f1_utils
		self.best_model_path = best_model_path
		self.batch_size = batch_size
		self.CLASS = CLASS

	def init_attributes(self):
		""""
		Set default values to some attributes.
		Args:
		Return:
		"""
		self.best_f1 = -1
		self.best_epoch = -1
		self.wait = 0

	def check_attributes(self):
		""""
		Check whether some required attributes have been set.
		If not, assert.
		Args:
		Return:
		"""
		attributes = [self.f1_utils, 
			self.best_model_path, 
			self.batch_size, 
			self.CLASS]

		for attribute in attributes:
			assert attribute is not None

	def init_train_conf(self):
		""""
		Initialize most of attribute values from the configuration file.
		Args:
		Return:

		"""	
		conf_dir = self.conf_dir
		train_cfg_path = os.path.join(conf_dir, 'train.cfg')
		assert os.path.exists(train_cfg_path)
		config = configparser.ConfigParser()
		config.read(train_cfg_path)

		assert 'metricCallback' in config.sections()
		train_conf = config['metricCallback']
		self.learning_rate = float(train_conf['learning_rate'])
		self.decay_rate = float(train_conf['decay_rate'])
		self.epoch_of_decay = int(train_conf['epoch_of_decay'])
		self.early_stop = int(train_conf['early_stop'])
		assert 'validate' in config.sections()
		vali_conf = config['validate']
		self.metric = vali_conf['metric']
		self.ave = vali_conf['ave']

	def get_at(self, preds, labels):
		""""
		Count audio tagging performance (F1).
		Args:
			preds: numpy.array
				shape: [number_of_files_( + padding), CLASS]
					prediction of the model
			labels: numpy.array
				shape: [number_of_files_( + padding), CLASS]
					labels loaded from files
		Return:	
			f1: float
			the audio tagging performance (F1)
		"""
		f1_utils = self.f1_utils
		f1, _, _, _, _, _ = f1_utils.get_f1(preds, labels, mode = 'at')
		return f1

	def get_opt(self, lr):
		""""
		Optimizer with specified learning rate.
		Args:
			lr: float
				learning rate
		Return:
			opt: keras.optimizers
				Adam optimizer
		"""
		opt = keras.optimizers.Adam(lr = lr, beta_1 = 0.9, 
			beta_2 = 0.999, epsilon = 1e-8, decay = 1e-8)
		return opt

	def get_loss(self):
		""""
		Loss function for semi-supervised learning and weakly-supervised learning.
		Args:
		Return:
			loss (if train_mode is 'supervised'): function
				loss function for weakly-supervised learning
			semi_loss (if train_mode is 'semi'): function
				loss function for semi-supervised learning
		"""

		CLASS = self.CLASS
		train_mode = self.train_mode

		def loss(y_true, y_pred):
			""""
			Loss function for weakly-supervised learning.

			"""
			return K.mean(K.binary_crossentropy(y_true[:, :CLASS], 
						y_pred), axis = -1)

		def semi_loss(y_true, y_pred):
			""""
			Loss function for semi-supervised learning.

			"""

			#weights of the unsupervised loss for the PT-model
			a = y_true[:, CLASS * 2:CLASS * 2 + 1]
			#label mask
			mask = y_true[:, CLASS:CLASS * 2]
			#groundtruth
			y_true = y_true[:, :CLASS]
			#the predictions (possibilities) of the PT-model
			y_pred_1 = y_pred[:, :CLASS]
			#the predictions (possibilities) of the PS-model
			y_pred_2 = y_pred[:, CLASS:]

			#the 0-1 predictions of the PS-model
			y_pred_2_X = K.relu(K.relu(y_pred_2, threshold = 0.5) * 2, 
					max_value = 1)

			#the 0-1 predictions of the PT-model
			y_pred_1_X = K.relu(K.relu(y_pred_1, threshold = 0.5) * 2, 
					max_value = 1)

			#the supervised loss for the PT-model with labeled data
			closs = K.mean(K.binary_crossentropy(
					y_true * mask, y_pred_1 * mask), axis = -1)
			#the supervised loss for the PS-model with labeled data
			closs += K.mean(K.binary_crossentropy(
					y_true * mask, y_pred_2 * mask), axis = -1)

			#mask for unlabeled data
			mask = 1-mask

			#the unsupervised loss for the PS-model
			closs += K.mean(K.binary_crossentropy(
				y_pred_1_X * mask, y_pred_2 * mask), axis = -1)
			#the unsupervised loss for the PT-model with weights a
			closs += K.mean(K.binary_crossentropy(
				y_pred_2_X * mask * a, y_pred_1 * mask * a), axis = -1)
			return closs

		if train_mode == 'supervised':
			return loss
		elif train_mode == 'semi':
			return semi_loss
		assert True

	def on_train_begin(self, logs = {}):
		""""
		(overwrite)
		The beginning of training.

		"""
		#check extra required attributes
		self.check_attributes()
		LOG.info('init training...')
		LOG.info('metrics : %s %s'%(self.metric, self.ave))

		opt = self.get_opt(self.learning_rate)
		loss = self.get_loss()
		#compile the model with specific loss function
		self.model.compile(optimizer = opt, loss = loss)
	
	def on_epoch_end(self, epoch, logs = {}):
		""""
		(overwrite)
		The end of a training epoch.

		"""
		best_f1 = self.best_f1
		f1_utils = self.f1_utils
		CLASS = self.CLASS
		train_mode = self.train_mode
		early_stop = self.early_stop

		#get the features of the validation data
		vali_data = self.validation_data
		#get the labels of the validation data
		labels = vali_data[1][:, :CLASS]

		#get audio tagging predictions of the model
		preds = self.model.predict(vali_data[0], batch_size = self.batch_size)
		
		if train_mode == 'semi':
			#get the predictions of the PT-model
			preds_PT = preds[:, :CLASS]
			#get the predictions of the PS-model
			preds_PS = preds[:, CLASS:]
			#count F1 score on the validation set for the PT-model
			pt_f1 = self.get_at(preds_PT, labels)
			#count F1 score on the validation set for the PS-model
			ps_f1 = self.get_at(preds_PS, labels)
		else:
			#count F1 score on the validation set for the PS-model
			ps_f1 = self.get_at(preds, labels)

		#the final performance depends on the PS-model
		logs['f1_val'] = ps_f1

		is_best = 'not_best'

		#preserve the best model during training
		if logs['f1_val'] >= self.best_f1:
			self.best_f1 = logs['f1_val']
			self.best_epoch = epoch
			self.model.save_weights(self.best_model_path)
			is_best = 'best'
			self.wait = 0

		#the PS-model has not been improved after [wait] epochs
		self.wait += 1

		#training early stops if there is no more improvement
		if self.wait > early_stop:
			self.stopped_epoch = epoch
			self.model.stop_training = True


		if train_mode == 'semi':
			LOG.info('[ epoch %d , sed f1 : %f , at f1 : %f ] %s'
				%(epoch, logs['f1_val'], pt_f1, is_best))
		else:
			LOG.info('[ epoch %d, f1 : %f ] %s'
				%(epoch, logs['f1_val'], is_best))

		#learning rate decays every epoch_of_decay epochs
		if epoch > 0 and epoch%self.epoch_of_decay == 0:
			self.learning_rate *= self.decay_rate
			opt = self.get_opt(self.learning_rate)
			LOG.info('[ epoch %d , learning rate decay to %f ]'%(
					epoch, self.learning_rate))
			loss = self.get_loss()
			#recompile the model with decreased learning rate
			self.model.compile(optimizer = opt, loss = loss)
		
		
	def on_train_end(self, logs = {}):
		""""
		(overwrite)
		The end of training.

		"""
		best_epoch = self.best_epoch
		best_f1 = self.best_f1
		#report the best performance of the PS-model
		LOG.info('[ best vali f1 : %f at epoch %d ]'%(best_f1, best_epoch))

	

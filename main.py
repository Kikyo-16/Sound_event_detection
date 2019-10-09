import copy
import os
import numpy as np
import random
import shutil
import sys
from src import trainer
from keras import backend as K
import tensorflow as tf
import argparse
from src.Logger import LOG
from keras.layers import Input,concatenate,GaussianNoise
from keras.models import load_model,Model
import shutil
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def supervised_train(task_name,sed_model_name,augmentation):
	""""
	Training with only weakly-supervised learning
	Args:
		task_name: string
			the name of the task
		sed_model_name:	string
			the name of the model
		augmentation:	bool
			whether to add Gaussian noise Layer
	Return:

	"""
	LOG.info('config preparation for %s'%sed_model_name)
	#prepare for training
	train_sed=trainer.trainer(task_name,sed_model_name,False)
	
	#creat model using the model structure prepared in [train_sed]
	creat_model_sed=train_sed.model_struct.graph()
	LEN=train_sed.data_loader.LEN
	DIM=train_sed.data_loader.DIM
	inputs=Input((LEN,DIM))

	#add Gaussian noise Layer
	if augmentation:
		inputs_t=GaussianNoise(0.15)(inputs)
	else:
		inputs_t=inputs
	outs=creat_model_sed(inputs_t,False)

	#the model used for training
	models=Model(inputs,outs)

	LOG.info('------------start training------------')
	train_sed.train(extra_model=models,train_mode='supervised')

	#predict results for validation set and test set
	train_sed.save_at_result()	#audio tagging result
	train_sed.save_sed_result()	#event detection result

def semi_train(task_name,sed_model_name,at_model_name,augmentation):
	""""
	Training with semi-supervised learning (Guiding learning)
	Args:
		task_name: string
			the name of the task
                sed_model_name: string
			the name of the the PS-model
		at_model_name: string
			the name of the the PT-model
                augmentation: bool
			whether to add Gaussian noise to the input of the PT-model
	Return:

        """
	#prepare for training of the PS-model
	LOG.info('config preparation for %s'%at_model_name)
	train_sed=trainer.trainer(task_name,sed_model_name,False)

	#prepare for training of the PT-model
	LOG.info('config preparation for %s'%sed_model_name)
	train_at=trainer.trainer(task_name,at_model_name,False)

	#connect the outputs of the two models to produce a model for end-to-end learning
	creat_model_at=train_at.model_struct.graph()
	creat_model_sed=train_sed.model_struct.graph()
	LEN=train_sed.data_loader.LEN
	DIM=train_sed.data_loader.DIM	
	inputs=Input((LEN,DIM))

	#add Gaussian noise
	if augmentation:
		at_inputs=GaussianNoise(0.15)(inputs)
	else:
		at_inputs=inputs

	at_out=creat_model_at(at_inputs,False)
	sed_out=creat_model_sed(inputs,False)
	out=concatenate([at_out,sed_out],axis=-1)
	models=Model(inputs,out)

	#start training (all intermediate files are saved in the PS-model dir)
	LOG.info('------------start training------------')	
	train_sed.train(models)

	#copy the final model to the PT-model dir from the PS-model dir
	shutil.copyfile(train_sed.best_model_path,train_at.best_model_path) 

	#predict results for validation set and test set (the PT-model)
	LOG.info('------------result of %s------------'%at_model_name)
	train_at.save_at_result()	#audio tagging result

	#predict results for validation set and test set (the PS-model)
	LOG.info('------------result of %s------------'%sed_model_name)
	train_sed.save_at_result()	#audio tagging result
	train_sed.save_sed_result()	#event detection result


def test(task_name, model_name, model_path = None, at_preds={}, sed_preds={}):
	""""
	Test with prepared model dir.
	The format of the model dir must be consistent with the required format.
	Args:
		task_name: string
			the name of the task
		model_name: string
			the name of the model
		model_path: string
			the path of model weights (if None, set defaults)
		at_preds: dict
		sed_preds: dict

	Return:
		at_preds: dict
			{'vali': numpy.array, 'test': numpy.array}
			audio tagging prediction (possibilities) on both set
		sed_preds: dict
			{'vali': numpy.array, 'test': numpy.array}
			detection prediction (possibilities) on both set
	
        """
	#prepare for testing
	train = trainer.trainer(task_name,model_name,True)
	if not model_path == None:
		train.best_model_path = model_path
	#predict results for validation set and test set
	at_preds_out = train.save_at_result(at_preds)	#audio tagging result
	sed_preds_out = train.save_sed_result(sed_preds)	#event detection result
	return at_preds_out, sed_preds_out

def bool_convert(value):
	""""
	Convert a string to a boolean type value
	Args:
		value: string in ['true','True','false','False']
			the string to convert
	Return:
		rvalue: bool
			a bool type value

	"""
	if value=='true' or value=='True':
		rvalue=True
	elif value=='false' or value=='False':
		rvalue=False
	else:
		assert False
	return rvalue


def test_models(task_name, model_name, model_list_path):
	""""
	Test with prepared model dir.
	The format of the model dir and model weights must be consistent with the required format.
	Args:
		task_name: string
			the name of the task
		model_name: string
			the name of the model
		model_list_path: string
			the path of file which keeps a list of paths of model weights
	Return:

	"""

	def predict(A):
		A[A >= 0.5 ] = 1
		A[A < 0.5] = 0
		return A

	if model_list_path == None:
		test(task_name,sed_model_name)
	else:
		with open(model_list_path) as f:
			model_list = f.readlines()
		model_list = [m.rstrip() for m in model_list]

		if len(model_list) == 1:
			LOG.info( 'ensemble results (just a single model)')
			test(task_name, sed_model_name, model_list[0])
			return
		at_results={}
		sed_results={}
		mode = ['vali', 'test']
		for model_path in model_list:
			LOG.info( 'decode for model : {}'.format(model_path))
			at_preds, sed_preds = test(task_name, sed_model_name, model_path)
			for m in mode:
				if m not in at_results:
					at_results[m] = predict(at_preds[m])
					sed_results[m] = sed_preds[m]
				else:
					at_results[m] += predict(at_preds[m])
					sed_results[m] += sed_preds[m]
			

		for m in mode:
			at = copy.deepcopy(at_results[m])

			#vote for boundary detection
			mask = np.reshape(at, [at.shape[0],1,at.shape[1]])
			mask[mask == 0] = 1
			sed_results[m] /= mask	

			#vote for audio tagging
			at_results[m] = at / len(model_list)
			sed_results[m] = [at_results[m], sed_results[m]]

		LOG.info( 'ensemble results')	
		test(task_name, sed_model_name, None, at_results, sed_results)
					

if __name__=='__main__':
	LOG.info('Disentangled feature')
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-n', '--task_name', 
			dest='task_name',
			help='task name')

	parser.add_argument('-s', '--PS_model_name', dest='PS_model_name',
		help='the name of the PS model')
	parser.add_argument('-t', '--PT_model_name', dest='PT_model_name',
		help='the name of the PT model')
	parser.add_argument('-md', '--mode', dest='mode',
		help='train or test')
	parser.add_argument('-g', '--augmentation', dest='augmentation',
		help='select [true or false] : whether to use augmentation (add Gaussian noise)')
	parser.add_argument('-u', '--semi_supervised', dest='semi_supervised',
		help='select [true or false] : whether to use unlabel data')
	parser.add_argument('-e', '--ensemble', dest='ensemble',
		help='select [true or false] : whether to ensembel several models when testing')
	parser.add_argument('-w', '--model_weights_list', dest='model_weights_list',
		help='the path of file containing a list of path of model weights to ensemble')
	f_args = parser.parse_args()

	task_name = f_args.task_name
	sed_model_name = f_args.PS_model_name
	at_model_name = f_args.PT_model_name
	mode = f_args.mode
	semi_supervised = f_args.semi_supervised
	augmentation = f_args.augmentation
	ensemble = f_args.ensemble
	model_weights_list = f_args.model_weights_list

	if mode not in ['train','test']:
		LOG.info('Invalid mode')
		assert LOG.info('try add --help to get usage')

	if task_name is None:
		LOG.info('task_name is required')
		assert LOG.info('try add --help to get usage')

	if sed_model_name is None:
		LOG.info('PS_model_name is required')		
		assert LOG.info('try add --help to get usage')

	if mode == 'train':
		augmentation = bool_convert(augmentation)
		semi_supervised = bool_convert(semi_supervised)
		if semi_supervised and at_model_name is None:
			LOG.info('PT_model_name is required for semi-supervised learning')
			assert LOG.info('try add --help to get usage')
	
			assert LOG.info('try add --help to get usage')
	else:
		semi_supervised = False
		ensemble = bool_convert(ensemble)
		if not ensemble:
			model_weights_list = None
		

	LOG.info( 'task name: {}'.format(task_name))
	LOG.info( 'PS-model name: {}'.format(sed_model_name))
	if semi_supervised:
		LOG.info( 'PT-model name: {}'.format(at_model_name))
	LOG.info( 'mode: {}'.format(mode))
	LOG.info( 'semi_supervised: {}'.format(semi_supervised))
	
	if mode=='train':
		if semi_supervised:
			semi_train(task_name, sed_model_name, at_model_name, augmentation)
		else:
			supervised_train(task_name, sed_model_name, augmentation)		
	else:
		test_models(task_name, sed_model_name, model_weights_list)
		


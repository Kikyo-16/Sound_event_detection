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


def test(task_name,model_name):
	""""
	Test with prepared model dir.
	The format of the model dir must be consistent with the required format.
	Args:
		task_name: string
			the name of the task
		model_name: string
			the name of the model
	Return:
	
        """
	#prepare for testing
	train=trainer.trainer(task_name,model_name,True)

	#predict results for validation set and test set
	train.save_at_result()	#audio tagging result
	train.save_sed_result()	#event detection result

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
	f_args = parser.parse_args()

	task_name=f_args.task_name
	sed_model_name=f_args.PS_model_name
	at_model_name=f_args.PT_model_name
	mode=f_args.mode
	semi_supervised=f_args.semi_supervised
	augmentation=f_args.augmentation

	augmentation=bool_convert(augmentation)
	semi_supervised=bool_convert(semi_supervised)

	if task_name is None:
		LOG.info('task_name is required')
		assert LOG.info('try add --help to get usage')

	if sed_model_name is None:
		LOG.info('PS_model_name is required')		
		assert LOG.info('try add --help to get usage')

	if semi_supervised and at_model_name is None:
		LOG.info('PT_model_name is required for semi-supervised learning')
		assert LOG.info('try add --help to get usage')
	
	if mode not in ['train','test']:
		LOG.info('Invalid mode')
		assert LOG.info('try add --help to get usage')


	LOG.info( 'task name: {}'.format(task_name))
	LOG.info( 'PS-model name: {}'.format(sed_model_name))
	if semi_supervised:
		LOG.info( 'PT-model name: {}'.format(at_model_name))
	LOG.info( 'mode: {}'.format(mode))
	LOG.info( 'semi_supervised: {}'.format(semi_supervised))
	
	if mode=='train':
		if semi_supervised:
			semi_train(task_name,sed_model_name,at_model_name,
				augmentation)
		else:
			supervised_train(task_name,sed_model_name,augmentation)		
	else:
		test(task_name,sed_model_name)
		


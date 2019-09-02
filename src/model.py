import os
import configparser
from keras import backend as K
import keras
import tensorflow as tf
from keras.layers import Layer
from keras.models import load_model, Model
from keras.layers import Permute, Reshape, Lambda, Bidirectional, Conv2DTranspose, dot
from keras.layers import Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Activation, BatchNormalization, TimeDistributed, Dropout
from keras.layers import GRU, Dense, Input, Activation, Conv2D, MaxPooling2D
from keras.layers import Dot, add, multiply, concatenate, subtract, GlobalMaxPooling1D
from keras.layers import UpSampling2D, GlobalMaxPooling2D
import numpy as np

 
class attentionLayer(Layer):
	def __init__(self, ** kwargs):
		""""
		Class-wise attention pooling layer
                Args:
                Attributes:
			kernel: tensor
			bias: tensor	

		"""
		super(attentionLayer, self).__init__( ** kwargs)

	def build(self, input_shape):

		kernel_shape = [1] * len(input_shape)
		bias_shape = tuple([1] * (len(input_shape)-1))
		
		kernel_shape[-1] = input_shape[-1]
		kernel_shape = tuple(kernel_shape)
		
		self.kernel = self.add_weight(
				shape = kernel_shape, 
				initializer = keras.initializers.Zeros(), 
				name = '%s_kernel'%self.name)

		self.bias = self.add_weight(
				shape = bias_shape, 
				initializer = keras.initializers.Zeros(), 
				name = '%s_bias'%self.name)

		super(attentionLayer, self).build(input_shape)
	
	def call(self, inputs):
		weights = K.sum(inputs * self.kernel, axis = -1) + self.bias
		return weights

	def compute_output_shape(self, input_shape):
		out_shape = []
		for i in range(len(input_shape)-1):
			out_shape += [input_shape[i]]
		return tuple(out_shape)

class sumLayer(Layer):
	""""
	Layer that sums tensor elements over a given axis.
	It takes as input a tensor.
	Args:
		axi: integer
			axis or axes along which a sum is performed
		keep_dims: bool
			if this is set to True, the axes which are reduced are left in the result as dimensions with size one
	Input shape:
	Output shape:

	"""
	def __init__(self, axi, keep_dims, ** kwargs):
		self.axi = axi
		self.keep_dims = keep_dims
		super(sumLayer, self).__init__( ** kwargs)

	def build(self, input_shape):
		super(sumLayer, self).build(input_shape)

	def call(self, inputs):
		axi = self.axi
		out = K.sum(inputs, axis = axi)
		out = K.expand_dims(out, axis = axi)
		return out

	def compute_output_shape(self, input_shape):
		axi = self.axi
		keep_dims = self.keep_dims
		if keep_dims:
			input_shape[axi] = 1
			return input_shape
		out_shape = []
		for i in range(len(input_shape)):
			if axi == i:
				continue
			out_shape += [input_shape[i]]
		return tuple(out_shape)



class attend_cnn(object):

	def __init__(self, conf_dir, model_name, task_name, LEN, DIM, CLASS):
		""""
		Build the model according to the configuration file.	
		Args:
			conf_dir: string
				the path of configuration dir
			model_name: string
				the name of the model
			task_name: string
				the name of the task
			LEN: integer
				the sequence length of the input feature
			DIM: integer
				the dimension of the input feature
			CLASS: integer
				the number of classes
				
		Attributes:
			conf_dir
			model_name
			task_name
			LEN
			DIM
			CLASS
			cnn_layer
			filters
			pool_size
			dilation_rate
			conv_nums
			kernel_size
			with_rnn
			(hid_dim)
			with_dropout
			(dr_rate)
			top_len
			dfs
			
		Interface:
			init_model_conf: Initialize most of attribute values from model configuration file.
			set_DFs: Set the coefficients for the DF dimension.
			cnn_block: Construct several CNN layers, several BN layers, a Maxpooling layer and a Dropout layer (optional) for a CNN block.
			apply_layers: Pass the input through a series of layers.
			get_CNN_blocks: Construct several CNN blocks for feature encoder.
			get_rnn_module: Construct RNN module which follows CNN block for feature encoder.
			get_attention_module: Construct class-wise attention pooling layers.
			get_classifier: Construct classifiers for clip-level classification.
			get_DF_dims: Count DF dimession per class.
			graph: A tool helping to construct the target model.
			get_model: Construct a model in under the specified mode. 

		"""	
		self.conf_dir = conf_dir
		self.model_name = model_name
		self.task_name = task_name
		self.LEN = LEN
		self.DIM = DIM
		self.CLASS = CLASS
		self.init_model_conf()
	
	def init_model_conf(self):
		""""
		Initialize most of attribute values from model configuration file.
		Args:
		Return:
		
		"""

		model_name = self.model_name
		conf_dir = self.conf_dir
		#the path of model configuration file
		model_cfg_path = os.path.join(conf_dir, 'model.cfg')

		assert os.path.exists(model_cfg_path)
		config = configparser.ConfigParser()
		config.read(model_cfg_path)
		assert model_name in config.sections()
		model_conf = config[model_name]

		self.pooling_mode = model_conf['pooling_mode']

		filters = model_conf['filters'].split(',')
		conv_num = model_conf['conv_num'].split(',')
		pool_size = model_conf['pool_size'].split(',')
		dilation_rate = model_conf['dilation_rate'].split(',')
		kernel_size = model_conf['kernel_size'].split(',')

		self.cnn_layer = int(model_conf['cnn_layer'])
		self.filters = [int(c) for c in filters]
		self.conv_num = [int(c) for c in conv_num]
		self.with_rnn = config.getboolean(model_name, 'with_rnn')
		self.with_dropout = config.getboolean(model_name, 'with_dropout')
		self.pool_size = [[int(p.split(' ')[0]), int(p.split(' ')[1])] 
					for p in pool_size]
		self.dilation_rate = [[int(p.split(' ')[0]), int(p.split(' ')[1])]
					for p in dilation_rate]
		self.kernel_size = [[int(k.split(' ')[0]), int(k.split(' ')[1])]
					for k in kernel_size]

		if self.with_rnn:
			self.hid_dim = int(model_conf['hid_dim'])
		else:
			self.hid_dim = 0

		if self.with_dropout:
			self.dr_rate = float(model_conf['dr_rate'])
		else:
			self.dr_rate = 0

		#count the sequence length and the dimession of the feature output by the featrue encoder
		top_len = self.LEN
		top_dim = self.DIM
		for i in range(self.cnn_layer):
			top_len //= self.pool_size[i][0]
			top_dim //= self.pool_size[i][1]
		top_dim *= self.filters[-1]
		self.top_len = top_len
		self.top_dim = top_dim


	def set_DFs(self, dfs):
		""""
		Set the coefficients for the DF dimension.
		coefficient x hidden feature dimension = DF dimension
		Args:
			dfs: list
				coefficients for the DF dimension count from the training set
		Return:

                """
		self.dfs = dfs

	def cnn_block(self, filters, kernel_size, name, conv_num, pool_size, 
				dilation_rate, dr_mode, dr_rate = None):
		""""
		Construct several CNN layers, several BN layers, a Maxpooling layer and a Dropout layer (optional) for a CNN block.
		Args:

			filters: integer
				the number of filters per CNN layer
			kernel_size: tuple or list
				the size of the filter
			name: string
				a unique identifier used to form the name of CNN layer
			conv_num: integer
				the number of CNN layers in a CNN block
			pool_size: tuple or list
				the pool size of the Maxpooling layer
			dilation_rate: tuple or list
				the dilation rate of the filter	
			dr_mode: bool
				whether there is a Dropout layer in a CNN block
			(dr_rate): float
				If dr_mode is true, then the dropout rate of the Dropout layer
		Return:
			cnns: list
				multiple layers
				conv_num x [Conv2D, BatchNormalization, Activation]
					 + [MaxPooling2D]( + [Dropout])

		"""
		cnns = []

		#model_name: to get a unique name for each layer when there are multiple models with CNN blocks
		md = self.model_name

		
		for i in range(conv_num):
			cnns += [Conv2D(filters = filters, 
					kernel_size = kernel_size, 
					dilation_rate = dilation_rate, 
					padding = 'same', 
					name = '%s_cnn_bl_conv_%d_%d'%(md, name, i)), 
				BatchNormalization(axis = -1, name = '%s_cnn_bl_bn_%d_%d'%(md, name, i)), 
				Activation('relu', 
					name = '%s_cnn_bl_ac_%d_%d'%(md, name, i))]
			
		mp = MaxPooling2D(pool_size = pool_size, 
			name = '%s_cnn_bl_mp_%d'%(md, name))
		dr = Dropout(rate = dr_rate, name = '%s_cnn_bl_dr_%d'%(md, name))

		cnns += [mp]
		if dr_mode:
			cnns += [dr]
		return cnns


	def apply_layers(self, inputs, layers):
		""""
		Pass the input through a series of layers.
		These layers must be continuous.
		Args:
			inputs: tensor
				shape: [batch_size, ...]
			layers: list
				multiple layers
			
		Return:
			out: tensor
				the output of the final layer

		"""
		out = inputs
		for layer in layers:
			out = layer(out)
		return out

	def get_CNN_blocks(self):
		""""
                Construct several CNN blocks for feature encoder.
		Args:
		Return:
			cnns: list
				multiple layers

		"""
		cnn_layer = self.cnn_layer
		filters = self.filters
		kernel_size = self.kernel_size
		conv_num = self.conv_num
		pool_size = self.pool_size
		dilation_rate = self.dilation_rate
		with_dropout = self.with_dropout
		dr_rate = self.dr_rate
		
		cnns = []
		for i in range(cnn_layer):
			cnns += self.cnn_block(filters[i], 
					kernel_size[i], 
					name = i, 
					conv_num = conv_num[i], 
					pool_size = pool_size[i], 
					dilation_rate = dilation_rate[i], 
					dr_mode = with_dropout, 
					dr_rate = dr_rate)
		return cnns

	def get_rnn_module(self):
		""""
                Construct RNN module which follows CNN block for feature encoder.
		Args:
		Return:
			cnns: list
				multiple layers
				[GRU, BatchNormalization, Activation(ReLU)]
		"""
		hid_dim = self.hid_dim
		md = self.model_name
		gru = GRU(hid_dim, name = '%s_GRU'%md, 
			return_sequences = True, dropout = 0.1)
		bc = BatchNormalization(axis = -1, name = '%s_gru_bc'%md)
		ac = Activation('relu', name = '%s_gru_ac'%md)
		grus = [gru, bc, ac]
		return grus

	def get_attention_module(self):
		""""
		Construct class-wise attention pooling layers.
		Arg:
		Return:
			attens: list
				multiple layers
				[attentionLayer] x CLASS
			
		"""
		CLASS = self.CLASS
		md = self.model_name
		attens = []
		for i in range(CLASS):
			attens += [attentionLayer(name = '%s_atten_%d'%(md, i))]
		return attens

	def get_classifier(self):
		""""
		Construct classifiers for clip-level classification.
		Arg:
		Return:
			denses: list
				multiple layers
				[denses] x CLASS

		"""
		CLASS = self.CLASS
		md = self.model_name
		denses = []
		for i in range(CLASS):
			denses += [Dense(1, use_bias = True, name = '%s_Dense_%d'%(md, i))]
		return denses

	def get_DF_dims(self, h_dim = None):
		""""
		Count DF dimession per class.
		Arg:
			h_dim: integer or None
				the dimession of the feature output by feature encoder
				if h_dim is None, it would be set to top_dim.
		"""
		CLASS = self.CLASS
		md = self.model_name
		dfs = self.dfs

		if h_dim == None:
			h_dim = self.top_dim

		hiddens = []
		for i in range(CLASS):
			hiddens += [int(h_dim * dfs[i])]
		return hiddens


	def graph(self):
		""""
		A tool helping to construct the target model.
		Args:
		Interface:
			feature_encoder: the feature encoder
			create_model: a tool helping to construct the target model
		
		Return:
			create_model: function
				a tool helping to construct the target model

		"""

		CLASS = self.CLASS
		LEN = self.LEN
		DIM = self.DIM
		top_len = self.top_len
		top_dim = self.top_dim
		with_rnn = self.with_rnn
		md = self.model_name
		pooling_mode = self.pooling_mode
		h_dim = None
		grus = []

		#get all the CNN blocks to construct feature encoder
		cnns = self.get_CNN_blocks()

		#If RNN is used, get RNN module to construct feature encoder
		#The dimession of the feature output by the feature encoder is no longer top_dim and it should be consistent with hid_dim
		if with_rnn:
			grus += get_rnn_module()
			h_dim = self.hid_dim

		#get the attention pooling layer per class
		attens = self.get_attention_module()
		#get the classifier per class
		denses = self.get_classifier()
		#get DF dimession per class
		hiddens = self.get_DF_dims(h_dim)

		#input BN layer
		inputs_BN = BatchNormalization(axis = -1, 
			name = '%s_BatchNormalization_input'%md)

		def feature_encoder(inputs):
			""""
			The feature encoder.
			Args:
				inputs: tensor
					shape: [batch_size, LEN, DIM]
			Return:
				out: tensor
					without rnn:
						shape: [batch_size, top_len, top_dim]
					with rnn:
						shape: [batch_size, top_len, hid_dim]
			"""
			layers = [Reshape([LEN, DIM, 1])] + cnns + \
				[Reshape([top_len, top_dim])] + grus
			out = self.apply_layers(inputs, layers)
			return out

		def create_model(inputs, test_mode = False):
			""""
			Pass the input feature through multiple layers and get the output of the model for tagging mode or detection mode.
			Args:
				inputs: tensor
					input feature of the model
\					shape: [batch_size, LEN, DIM]
				test_mode: bool
					If test_mode is true, construct model for tagging mode and audio_tag_out would be returned; and vice, construct model for detection mode and a tuple of audio_tag_out and detection_out would be returned.
			Return:
				audio_tag_out: tensor
					output (audio tagging) of the model
					output in both modes
					shape: [batch_size, CLASS]
				(detection_out): tensor
					output (event detection) of the model
					output only in detection mode
					shape: [batch_size, top_len, CLASS]
			"""

			inputs_bn = inputs_BN(inputs)

			#X = {x_1, ..., x_T}
			X = feature_encoder(inputs_bn)
	
			audio_tag_outs = []
			detection_outs = []	
			
			for c in range(CLASS):
				#kc
				k = hiddens[c]
				#x_c = {x_c1, ..., x_cT}
				x_c = Lambda(lambda x : x[:, :, :k])(X)

				#different pooling mode
				if pooling_mode == 'GAP' or pooling_mode == 'GMP':
					if pooling_mode == 'GAP':
						h = GlobalAveragePooling1D()(x_c)	
					else:
						h = GlobalMaxPooling1D()(x_c)
					detection_out = denses[c](x_c)
					detection_out = Activation('sigmoid')(
								detection_out)
						
				elif pooling_mode == 'cATP' or pooling_mode == 'GSP':
					if pooling_mode == 'cATP':
						w_x = attens[c](x_c)
					else:
						w_x = denses[c](x_c)
					w_x = Reshape((top_len, 1))(w_x)
					#attention output for frame-level prediction
					detection_out = Activation('sigmoid')(w_x)
					#a_c = softmax((W x x_c) / d)
					w_x = Reshape((top_len, ))(w_x)
					w_x_d = Lambda(lambda x:x / k)(w_x)
					a = Activation('softmax')(w_x_d)
					a = Reshape((top_len, 1))(a)
					#h_c = \sum (a_c x x_c)
					h = multiply([a, x_c])
					h = sumLayer(1, False)(h)
					h = Reshape((k, ))(h)

				#output for clip-level prediction
				audio_tag_out = denses[c](h)
				audio_tag_out = Activation('sigmoid')(audio_tag_out)

				detection_out = Lambda(lambda x:x, 
					name = '%s_frame_output_%d'%(md, c))(
								detection_out)
				
				detection_outs += [detection_out]
				audio_tag_outs += [audio_tag_out]

			#integrate CLASS categories
			audio_tag_out = concatenate(audio_tag_outs, axis = -1)
			detection_out = concatenate(detection_outs, axis = -1)

			if test_mode:
				return [audio_tag_out, detection_out]
			return audio_tag_out
	
		return create_model


	
	def get_model(self, pre_model = None, mode = 'at'):
		""""
		Construct a model in under the specified mode.
		if pre_model is none, return the initialized model;
		or load the weights onto the model from file or another model object
		Args:
			pre_model: None or string or Model
				the path of a model object file or a model object
			mode: string in ['at','sed']
				at: the model has a output for audio tagging
				sed: the model has a tuple of outputs for audio tagging and event detection
		Return:
			model: Model
				the target model

		"""
		#if the value of mode is invalid
		assert mode == 'at' or mode == 'sed'

		LEN = self.LEN
		DIM = self.DIM
		CLASS = self.CLASS

		#input layer
		inputs = Input((LEN, DIM))
		#get model creation tool
		creat_model = self.graph()
		#get the output of the model
		out = creat_model(inputs, mode == 'sed')
		#construct the model
		model = Model(inputs, out)

		#load the model from file or object
		if pre_model is not None:
			if type(pre_model) is str:
				model.load_weights(pre_model, by_name = True)
			else:
				tmp_path = '.%s-%s.h5'%(self.task_name, self.model_name)
				pre_model.save_weights(tmp_path)
				model.load_weights(tmp_path, by_name = True)
				os.remove(tmp_path)
		return model

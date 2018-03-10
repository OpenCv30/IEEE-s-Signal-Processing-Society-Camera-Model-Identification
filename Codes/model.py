import numpy as np
import cv2
import pdb

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D,Flatten
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.regularizers import l2
from keras import backend as K
ROW_AXIS =1; #Height
COL_AXIS=2; #width
CHANNEL_AXIS=3; #channel

#https://github.com/raghakot/keras-resnet/blob/master/resnet.py
class ResNetModel():
	def __init__(self,img_size):
		self.input_shape=img_size;
		self.channel_last_first= 'channels_last';
		self.filter_size_first_lyr = 64;
		self.kernel_size_first_lyr = 7;
		self.kernel_size_general =3;
		self.stride_size_first_lyr = 2;
		self.num_lyrs = 'LAYER_18';
		self.Model = self.build_model();
		print('Model is created. Model Name: {}'.format(self.num_lyrs));
	
	def residual_block(self,input_tensor,num_filter,kernel_size=(3,3),stride_size=(2,2),stage_name='x.x'):
		name_stage_bn='Bn_'+stage_name;
		name_stage_conv='Conv_'+stage_name;
		connection=input_tensor;
		out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name=name_stage_bn)(input_tensor);
		out = Activation('relu')(out)
		out= Conv2D(filters=num_filter, 
						kernel_size= kernel_size, 
						strides=stride_size, 
						padding='same', 
						data_format=self.channel_last_first, 
						kernel_initializer='he_normal', bias_initializer='zeros', 
						kernel_regularizer=l2(1.e-4),name=name_stage_conv)(out)
						
		return out;
	#@staticmethod		
	def build_model(self):
	#first Layer	
		input = Input(self.input_shape);
		#input = ZeroPadding2D((3, 3))(input)
		out= Conv2D(filters=self.filter_size_first_lyr, 
						kernel_size= (self.kernel_size_first_lyr,self.kernel_size_first_lyr), 
						strides=(self.stride_size_first_lyr,self.stride_size_first_lyr), 
						padding='same', 
						data_format=self.channel_last_first, 
						kernel_initializer='he_normal', bias_initializer='zeros', 
						kernel_regularizer=l2(1.e-4),name='Conv_First')(input)		
		x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name='Bn_First')(out);
		x = Activation('relu')(x)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same', data_format=self.channel_last_first)(x);
	 
	#Repetitive layers-#New improve method-{bn->relu-conv}
	#This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
		if( self.num_lyrs == 'LAYER_18'):
			repetitions=[2,2,2,2];
			Filters=[64,128,256,512];
			
			for i,r in enumerate(repetitions): # for Conv_block
				num_filter= int(Filters[i]);
				kernel_size = (self.kernel_size_general,self.kernel_size_general);
				is_first_block = True if(i==0) else False;
				for j in range(r): #within block
					stride =(1,1) if(is_first_block) else (2,2) if j==0 else (1,1)
					stage_name = str(i+1)+'_'+str(j+1)+'_a';
					direct_connection =x;
					if i==0 and j==0: #first layer of first conv block
						conv1 = Conv2D(filters=num_filter, 
									kernel_size=  kernel_size,
									strides=stride, 
									padding='same', 
									data_format=self.channel_last_first, 
									kernel_initializer='he_normal', bias_initializer='zeros', 
									kernel_regularizer=l2(1.e-4),name='Conv_'+stage_name)(x)
					else:
						conv1= self.residual_block(x,num_filter=num_filter,kernel_size=kernel_size,stride_size=stride,stage_name=stage_name)
					stage_name = str(i+1)+'_'+str(j+2)+'_b';
					stride=(1,1);
					residual= self.residual_block(conv1,num_filter=num_filter,kernel_size=kernel_size,stride_size=stride,stage_name=stage_name)
					
					x= self.residual_connection(direct_connection, residual)
	
		
			# Last activation
			x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name='Bn_Last_0')(x);
			block = Activation('relu')(x)

			# Classifier block
			block_shape = K.int_shape(block)
			pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
								 strides=(1, 1))(block)
			flatten1 = Flatten()(pool2) #num_nodes=512
			
			#dense1 = Dense(units=256, kernel_initializer="he_normal",
			#		  activation="softmax")(flatten1)
			dense = Dense(units=10, kernel_initializer="he_normal",
					  activation="softmax")(flatten1)

			model = Model(inputs=input, outputs=dense)
			return model;
			
	def residual_connection(self,directInput,residual):
		input_shape = K.int_shape(directInput)
		residual_shape = K.int_shape(residual)
		global ROW_AXIS,COL_AXIS,CHANNEL_AXIS
		#pdb.set_trace()
		stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
		stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
		equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

		shortcut = directInput
		# 1 X 1 conv if shape is different. Else identity.
		if stride_width > 1 or stride_height > 1 or not equal_channels:
			#pdb.set_trace()
			shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
						  kernel_size=(1, 1),
						  strides=(stride_width, stride_height),
						  padding="valid",
						  kernel_initializer="he_normal",
						  kernel_regularizer=l2(0.0001))(directInput)

		return layers.add([shortcut, residual])	   

	
		
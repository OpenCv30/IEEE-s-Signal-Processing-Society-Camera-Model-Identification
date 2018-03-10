import numpy as np	# linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import keras 
import tensorflow as tf
import zipfile
import pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pdb
import os

from keras.utils import plot_model
from model import ResNetModel
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint,TensorBoard, CSVLogger, ReduceLROnPlateau, EarlyStopping
from DataGenerator import DataGeneration
#from DataGeneratorAndDataAug import DataGeneration

''' OWN callback 
	Added Call back -modified to dump on every batch
	C:\mySoftware\python35\Lib\site-packages\keras\callbacks.py
'----------------------------------'''
'''LOADING DATA'''
trainDataFolderPath = r'C:\Machinelearning\cameraModel\data\train'
modelSaveFolderpath = r'C:\Machinelearning\cameraModel\models'
savePathcesTrain = r'C:\Machinelearning\cameraModel\trainsample\train'
savePathcesValidation = r'C:\Machinelearning\cameraModel\trainsample\valid'
num_patch_to_save = 2; #per batch


'''Training params'''		   
#Image= HxWxC
RANDOM_SEED=251;
NUMBER_EPOC = 100;
NUMBER_BATCH_PER_EPOC = 12;
img_width= 64;
img_height =64;
img_channel =3;
batch_size=256; #;
batch_size_valid= 256; #batch_size;
num_patch_to_validate = 2;
PARTIAL_TRAIN =1;


''' Load Model'''
input_image_shape= (img_height,img_width,img_channel);
resnet_object = ResNetModel(input_image_shape);


''' Save Model in Image'''
plot_model(resnet_object.Model,show_shapes =True, to_file='./model_train_dropout.png');
''' take prv trained model'''
if PARTIAL_TRAIN :
	weight_path = r'C:\Machinelearning\cameraModel\models\weights_sgd__Loaded_Weight_29_12_2017_. 12-1.946.hdf5';
	resnet_object.Model.load_weights(weight_path); #load_model('path_to_model);
	print('Model Loded from Pre_trained Weights!');

''' Compile Model'''
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
#adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.001)
resnet_object.Model.compile(optimizer=sgd,
							loss='categorical_crossentropy',
							metrics=['accuracy']);

#DATA AUGUMENTATION-
train_datagen = DataGeneration(
		trainDataPath=trainDataFolderPath,
		flip_H=True,
		flip_V=True,
		Rotation=True,
		JPEG_QF=True,
		RESIZE_BC=True,
		GAMMA_CORR=True,
		Rescale_Intensity=False,
		rescale=1./255,
		data_format='channels_last');
train_dir_generator = train_datagen.dataGeneratefromDirectory(
		data_dict_camera_image = train_datagen.trainDataDict,
		patch_size=(img_height,img_width),
		color_mode='rgb',
		num_patch=batch_size,
		num_class = 10,
		random_seed=RANDOM_SEED,
		class_label=None, #will be updated while reading
		save_data= True,
		num_patch_to_save = num_patch_to_save, #SAVE_ALL, for how many batch-images = num_patch_to_save*batch_size;
		save_to_dir= savePathcesTrain,
		save_prefix='Train_Aug',
		save_format='png');
		
''' Validation '''
valid_datagen = DataGeneration(
		trainDataPath=trainDataFolderPath,
		flip_H=True,
		flip_V=True,
		Rotation=True,
		JPEG_QF=True,
		RESIZE_BC=True,
		GAMMA_CORR=True,
		Rescale_Intensity=False,
		rescale=1./255,
		data_format='channels_last');
	
valid_dir_generator = valid_datagen.dataGeneratefromDirectory(
		data_dict_camera_image = valid_datagen.validationDataDict,
		patch_size=(img_height,img_width),
		color_mode='rgb',
		num_patch=batch_size_valid,
		num_class = 10,
		random_seed=RANDOM_SEED,
		class_label=None, #will be updated while reading
		save_data= True,
		num_patch_to_save = num_patch_to_save, #SAVE_ALL
		save_to_dir= savePathcesValidation,
		save_prefix='Valid_Aug',
		save_format='png');
	
''' Checkpoints '''
checkpoint = ModelCheckpoint('./models/weights_Adadelta__29_12_2017_.{epoch:03d}-{loss:0.3f}.hdf5')
tensorBoardcb = TensorBoard(log_dir="./TFlogs/adadelta", write_graph=True, write_images=True)
learningRatecb = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.00001, cooldown=1, min_lr=0.0001)
#terminateNancb = TerminateOnNaN()
#earlyStoppingcb = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=5, verbose=1, mode='auto')


''' Training '''
history2 = resnet_object.Model.fit_generator(train_dir_generator,
					steps_per_epoch= NUMBER_BATCH_PER_EPOC,
					epochs=NUMBER_EPOC,
					validation_data = valid_dir_generator,
					validation_steps = num_patch_to_validate,
					callbacks=[checkpoint,learningRatecb,tensorBoardcb],
					verbose=1)
resnet_object.Model.save('./camera_model__AdaDelta_Resnet18_29_12_2017.hdf5');
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

'''LOADING DATA'''
myTopDataFolderPath = r'C:\Machinelearning\cameraModel\data'
trainDataFolderPath = r'C:\Machinelearning\cameraModel\data\train'
modelSaveFolderpath = r'C:\Machinelearning\cameraModel\models'


'''Training params'''		   
#Image= HxWxC
RANDOM_SEED=251;
NUMBER_EPOC = 10;
NUMBER_BATCH_PER_EPOC = 512;
img_width= 64;
img_height =64;
img_channel =3;
batch_size=32;



''' Load Model'''
input_image_shape= (img_height,img_width,img_channel);
resnet_object = ResNetModel(input_image_shape);


''' Save Model in Image'''
plot_model(resnet_object.Model,show_shapes =True, to_file='./model_train_dropout.png');

''' Compile Model'''
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
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
	Rescale_Intensity=True,
	rescale=1./255,
	data_format='channels_last');

train_dir_generator = train_datagen.dataGeneratefromDirectory(
		patch_size=(img_height,img_width),
		color_mode='rgb',
		num_patch=batch_size,
		num_class = 10,
		random_seed=RANDOM_SEED,
		class_label=None, #will be updated while reading
		save_data= False,
		save_to_dir= r'C:\Machinelearning\cameraModel\trainsample\aug',
		save_prefix='Train_New_Aug',
		save_format='png');
	
''' Checkpoints '''
checkpoint = ModelCheckpoint('./models/weights_sgd_28_12_2017_.{epoch:04d}-{loss:.2f}.hdf5')
#checkpointer = ModelCheckpoint(modelSaveFolderpath+'\camera_model_Resnet18_best_only.hdf5', verbose=1, save_best_only=True)
tensorBoardcb = TensorBoard(log_dir="./TFlogs", write_graph=True, write_images=True)
#learningRatecb = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.00001, cooldown=1, min_lr=0.0001)
#terminateNancb = TerminateOnNaN()
#earlyStoppingcb = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=5, verbose=1, mode='auto')

''' Training '''
history2 = resnet_object.Model.fit_generator(train_dir_generator,
					steps_per_epoch= NUMBER_BATCH_PER_EPOC,
					epochs=NUMBER_EPOC,
					callbacks=[checkpoint,tensorBoardcb],
					verbose=1)
resnet_object.Model.save('./camera_model_DataAug_SGD_Resnet18.h5');
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

'''LOADING DATA'''
myTopDataFolderPath = r'C:\Machinelearning\cameraModel\data'
modelSaveFolderpath = r'C:\Machinelearning\cameraModel\models'
list_camera_folder=[]
DEBUG=1;
#check train directory
dataDict_cameras_imgPaths_train= dict();
for root, dirs, files in os.walk(myTopDataFolderPath+'\\train'):	
	for camera_model in dirs:
		for r,d,files in os.walk(myTopDataFolderPath+'\\train\\'+camera_model):
			if camera_model not in dataDict_cameras_imgPaths_train.keys():
				dataDict_cameras_imgPaths_train[camera_model]=[];
			for imgFile in files:
				file_path = myTopDataFolderPath+'\\train\\'+camera_model+'\\'+imgFile;
				dataDict_cameras_imgPaths_train[camera_model].append(file_path);
				
			
list_camera_folder = list(dataDict_cameras_imgPaths_train.keys());
if DEBUG:
	for mykey in dataDict_cameras_imgPaths_train.keys():
		print(mykey, len(dataDict_cameras_imgPaths_train[mykey]))

print (list_camera_folder)




'''Training params'''		   
#Image= HxWxC
RANDOM_SEED=251;
NUMBER_EPOC = 16;
NUMBER_BATCH_PER_EPOC = 1000;
img_width= 224;#64;
img_height =224;64;
img_channel =3;
batch_size=32;
Camera_Model=list_camera_folder; 
train_data_dir = myTopDataFolderPath+'\\train\\'
''' Load Model'''
input_image_shape= (img_height,img_width,img_channel);
resnet_object = ResNetModel(input_image_shape);


''' Save Model in Image'''
plot_model(resnet_object.Model,show_shapes =True, to_file='./model_train_!8.png');
pdb.set_trace()
''' Compile Model'''
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.01)
resnet_object.Model.compile(optimizer=adadelta,
							loss='categorical_crossentropy',
							metrics=['accuracy']);

#DATA AUGUMENTATION-
train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	vertical_flip= True,
	horizontal_flip=True,
	data_format= 'channels_last')

train_dir_generator = train_datagen.flow_from_directory(
	train_data_dir,
	target_size=(img_height,img_width),
	color_mode='rgb',
	batch_size=batch_size,
	classes = Camera_Model,
	seed=RANDOM_SEED,
	class_mode='categorical',
	save_to_dir= r'C:\Machinelearning\cameraModel\trainsample',
	save_prefix='Train_',
	save_format='png')

''' Checkpoints '''
#checkpointer = ModelCheckpoint(modelSaveFolderpath+'\camera_model_Resnet18_best_only.hdf5', verbose=1, save_best_only=True)
tensorBoardcb = TensorBoard(log_dir="./TFlogs", write_graph=True, write_images=True)
#learningRatecb = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.00001, cooldown=1, min_lr=0.0001)
#terminateNancb = TerminateOnNaN()
#earlyStoppingcb = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=5, verbose=1, mode='auto')

''' Training '''
history2 = resnet_object.Model.fit_generator(train_dir_generator,
					steps_per_epoch= NUMBER_BATCH_PER_EPOC,
					epochs=NUMBER_EPOC,
					callbacks=[tensorBoardcb],
					verbose=1)
resnet_object.Model.save_weights('./camera_model_Resnet18.h5');
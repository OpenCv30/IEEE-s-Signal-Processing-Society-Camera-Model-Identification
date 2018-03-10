import tensorflow as tf
import zipfile
import pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import pdb,time
import os
import numpy as np

from keras.utils import plot_model
from model import ResNetModel
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import ModelCheckpoint,TensorBoard, CSVLogger, ReduceLROnPlateau, EarlyStopping

import threading 
import queue 
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

from ProducerConsumer import Producer,BatchDataServer
from DataGenrationThreaded import DataGenerationThreaded

'''	Define Queue	'''

Max_Size =1024;
dataQueue = queue.Queue(Max_Size);

'''	LOADING DATA	'''
myTopDataFolderPath = r'C:\Machinelearning\cameraModel\data'
trainDataFolderPath = r'C:\Machinelearning\cameraModel\data\train'
modelSaveFolderpath = r'C:\Machinelearning\cameraModel\models'
save_patches= r'C:\Machinelearning\cameraModel\trainsample\gen'


'''	Training Params		'''		   
#Image= HxWxC
RANDOM_SEED=251;
NUMBER_EPOC = 10;
NUMBER_BATCH_PER_EPOC = 512;
patch_width= 64;patch_height =64;
img_channel =3;
batch_size=32;

''' Load Model'''
input_image_shape= (patch_height,patch_width,img_channel);
resnet_object = ResNetModel(input_image_shape);


''' Save Model in Image'''
plot_model(resnet_object.Model,show_shapes =True, to_file='./model_train.png');

''' Compile Model'''
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
#adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.001)
resnet_object.Model.compile(optimizer=sgd,
							loss='categorical_crossentropy',
							metrics=['accuracy']);
							

producerData_1= Producer(dataQueue,typeName='Producer_1',
				trainDataPath=trainDataFolderPath,
				patch_size=(patch_height,patch_width),
				batch_size = batch_size,
				num_class = 10,
				shuffle=True,
				flip_H=True,
				flip_V=True,
				Rotation=True,
				JPEG_QF=True,
				RESIZE_BC=True,
				GAMMA_CORR=True,
				shear_range=0.0,
				Rescale_Intensity=True,
				rescale=1./255,
				save_data=True,
				save_to_dir= save_patches,
				save_prefix='Train_P',
				save_format='png',
				data_format='channels_last');
				
producerData_2= Producer(dataQueue,typeName='Producer_2',
				trainDataPath=trainDataFolderPath,
				patch_size=(patch_height,patch_width),
				batch_size = batch_size,
				num_class = 10,
				shuffle=True,
				flip_H=True,
				flip_V=True,
				Rotation=True,
				JPEG_QF=True,
				RESIZE_BC=True,
				GAMMA_CORR=True,
				shear_range=0.0,
				Rescale_Intensity=True,
				rescale=1./255,
				save_data=True,
				save_to_dir= save_patches,
				save_prefix='Train_P',
				save_format='png',
				data_format='channels_last');
producerData_3= Producer(dataQueue,typeName='Producer_3',
				trainDataPath=trainDataFolderPath,
				patch_size=(patch_height,patch_width),
				batch_size = batch_size,
				num_class = 10,
				shuffle=True,
				flip_H=True,
				flip_V=True,
				Rotation=True,
				JPEG_QF=True,
				RESIZE_BC=True,
				GAMMA_CORR=True,
				shear_range=0.0,
				Rescale_Intensity=True,
				rescale=1./255,
				save_data=True,
				save_to_dir= save_patches,
				save_prefix='Train_P',
				save_format='png',
				data_format='channels_last');

getBatchData = BatchDataServer(dataQueue,
				num_patches=32,num_class=10,
				patch_size=(64,64),
				typeName='Main_Consumer');

producerData_1.setDaemon(True)
producerData_2.setDaemon(True)
producerData_3.setDaemon(True)
producerData_1.start();
producerData_2.start();
producerData_3.start();
'''	DATA AUGUMENTATION	'''
train_datagen = DataGenerationThreaded(dataGenMethodObject=getBatchData);
train_dir_generator = train_datagen.dataGeneratefromDirectory();
	
''' Checkpoints '''
checkpoint = ModelCheckpoint('../models/weights_Thread.{epoch:04d}-{loss:.2f}.hdf5',verbose =1)
tensorBoardcb = TensorBoard(log_dir="./TFlogs", write_graph=True, write_images=True)
#learningRatecb = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.00001, cooldown=1, min_lr=0.0001)
#terminateNancb = TerminateOnNaN()
#earlyStoppingcb = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=5, verbose=1, mode='auto')
csv_logger = CSVLogger('./Resnet18_DataAug_Thd.csv')

''' Training '''
history2 = resnet_object.Model.fit_generator(train_dir_generator,
					steps_per_epoch= NUMBER_BATCH_PER_EPOC,
					epochs=NUMBER_EPOC,
					callbacks=[checkpoint,tensorBoardcb,csv_logger],
					verbose=1)
resnet_object.Model.save('./camera_model_DataAug_SGD_Resnet18.h5');
producerData.join();
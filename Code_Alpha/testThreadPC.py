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
import threading 
import queue 
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

from ProducerConsumer import Producer,BatchDataServer
Max_Size =1024;
dataQueue = queue.Queue(Max_Size);

'''LOADING DATA'''
myTopDataFolderPath = r'C:\Machinelearning\cameraModel\data'
trainDataFolderPath = r'C:\Machinelearning\cameraModel\data\train'
modelSaveFolderpath = r'C:\Machinelearning\cameraModel\models'
save_patches= r'C:\Machinelearning\cameraModel\trainsample\gen'


'''Training params'''		   
#Image= HxWxC
RANDOM_SEED=251;
NUMBER_EPOC = 10;
NUMBER_BATCH_PER_EPOC = 512;
patch_width= 64;patch_height =64;
img_channel =3;
batch_size=32;

producerData= Producer(dataQueue,typeName='Producer_1',
		trainDataPath=trainDataFolderPath,
		patch_size=(patch_height,patch_width),
		batch_size = batch_size,
		num_class = 10,
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
			num_patches=batch_size,
			typeName='Main_Consumer');

producerData.start();



	


import tensorflow as tf
import zipfile
import pdb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import pdb
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

class DataGenerationThreaded():
	def __init__(self, dataGenMethodObject=None):
		
		self.getBatchDataObject = dataGenMethodObject;
		
		'''
		self.dataDict_cameras_imgPaths= dict();
		self.classLabel=[]; #determine from train folder data set
		
		self.readData2populateDict();
		''' '''To encode 10-class into Binary vector''''''
		self.LE= LabelEncoder();
		if self.classLabel is not None:
			classlen= len(self.classLabel);
			self.LE.fit(self.classLabel);
		else:
			assert(False),"One hot encoding is not done";
		'''
	def readData2populateDict(self):
		for root, dirs, files in os.walk(self.trainDataPath):	
			for camera_model in dirs:
				for r,d,files in os.walk(self.trainDataPath+'\\'+camera_model):
					if camera_model not in self.dataDict_cameras_imgPaths.keys():
						self.dataDict_cameras_imgPaths[camera_model]=[];
					for imgFile in files:
						file_path = self.trainDataPath+'\\'+camera_model+'\\'+imgFile;
						self.dataDict_cameras_imgPaths[camera_model].append(file_path);
		
		self.classLabel = list(self.dataDict_cameras_imgPaths.keys());
	
		
	def dataGeneratefromDirectory(self):
		color_mode ='rgb'
		if color_mode =='rgb':
			n_chn =3;
		else:
			n_chn =1;
		num_patch= self.getBatchDataObject.num_patches
		patch_size = self.getBatchDataObject.patch_size;
		rescale = self.getBatchDataObject.rescale;
		num_class = self.getBatchDataObject.num_class;
		batch_features = np.zeros((num_patch, patch_size[0], patch_size[1], n_chn),dtype=float) #bxhxwxc
		while True:
			batch_labels = [];
			data_label =self.getBatchDataObject.get_batch_data();
			for i in range(len(data_label)):
				batch_features[i]= np.float32(data_label[i][0])*rescale;
				batch_labels.append(data_label[i][1]);
			#pdb.set_trace()
			encoded_labels= list(batch_labels) #self.LE.transform(list(batch_labels))
			cat_label_matrix = to_categorical(encoded_labels,num_classes=num_class);
			#pdb.set_trace()
			yield batch_features,cat_label_matrix
			
			
		
	
	
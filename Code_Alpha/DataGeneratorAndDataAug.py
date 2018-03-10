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

from DataAugUtils import data_augments

ImplementedDataAugMethods = list(['NoChange','Hflip','Vflip','R90','R270',
					   'JPEG_QF_70','JPEG_QF_90',
					   'RESIZE_BC_0p5','RESIZE_BC_0p8','RESIZE_BC_1p5','RESIZE_BC_2p0',
						'GammaCorr_0p8','GammaCorr_1p2']);

class DataGeneration():
	def __init__(self,
		trainDataPath=r'C:\Machinelearning\cameraModel\data\train',
		flip_H=True,
		flip_V=True,
		rescale=1./255,
		data_format= 'channels_last'):
		
		self.trainDataPath = trainDataPath;
		self.flipH = flip_H;
		self.flipV = flip_V;
		self.rescale=rescale;
		self.num=2;
		self.dataDict_cameras_imgPaths= dict();
		self.classLabel=[]; #determine from train folder data set
		
		self.readData2populateDict();
		''' To encode 10-class into Binary vector'''
		self.LE= LabelEncoder();
		if self.classLabel is not None:
			classlen= len(self.classLabel);
			self.LE.fit(self.classLabel);
		else:
			assert(False),"One hot encoding is not done";
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
	
		
	def dataGeneratefromDirectory(self,
		patch_size=(64,64),
		num_patch= 32,
		num_class= 10,
		color_mode='rgb',
		class_label=None,
		save_data= True,
		save_to_dir= r'C:\Machinelearning\cameraModel\trainsample\gen',
		save_prefix='Train_',
		save_format='png',
		random_seed = 251):
		np.random.seed(seed=random_seed);
		if color_mode =='rgb':
			n_chn =3;
		else:
			n_chn =1;
		batch_features = np.zeros((num_patch, patch_size[0], patch_size[1], n_chn),dtype=np.float32) #bxhxwxc
		
		while True:
			batch_labels = [];
			data_label =self.patch_generation(patch_size,num_patch,num_class,self.dataDict_cameras_imgPaths,save_data,
							save_to_dir,save_prefix,save_format);
			for i in range(len(data_label)):
				batch_features[i]= (np.float32(data_label[i][0]))* self.rescale;
				batch_labels.append(data_label[i][1]);
			
			encoded_labels= self.LE.transform(list(batch_labels))
			cat_label_matrix = to_categorical(encoded_labels,num_classes=num_class);
			yield batch_features,cat_label_matrix
			
	
	
	@staticmethod
	def patch_generation(patch_size,num_patch,num_class,image_dict,
							save_data=False,save_to_dir= None,save_prefix='x.x',save_format='png'):
		
		assert (num_class == len(image_dict.keys())),"Difference in the number of classes between Image Dict and Input Argument of num_class = {}".format(num_class)
		patch_from_each_class = int(num_patch / num_class);
		random_patch_left = int(num_patch % num_class)
		
		p_h,p_w=patch_size; #hxw
		p_h_half = int(p_h/2)
		p_w_half = int(p_w/2)
		patch_image_label=[];image_name=[];
		for camera_model in image_dict.keys():
			num_images = len(image_dict[camera_model]);
			if( patch_from_each_class < num_images):
				randomIndexList = np.random.randint(0,num_images,patch_from_each_class);
				num_patch_per_class_image=1;
				for index in randomIndexList:
					image_name_with_path = image_dict[camera_model][index];
					img = cv2.imread(image_name_with_path);
					aug_method = np.random.choice(ImplementedDataAugMethods);
					
					aug_method_patchImage_flag = True;
					if aug_method  in [ 'JPEG_QF_70','JPEG_QF_90','RESIZE_BC_0p5','RESIZE_BC_0p8','RESIZE_BC_1p5','RESIZE_BC_2p0']:
						img = data_augments(aug_method,image_name_with_path)
						aug_method_patchImage_flag = False;
						
					H,W,C = img.shape;
					randomX = np.random.randint(p_w_half+1,W-p_w_half-1,num_patch_per_class_image);
					randomY = np.random.randint(p_h_half+1,H-p_h_half-1,num_patch_per_class_image);
					for X,Y in zip(randomX,randomY):
						patch_image= img[Y-p_h_half:Y+p_h_half,X-p_w_half:X+p_w_half,:]
						
						if aug_method_patchImage_flag:
							patch_image = data_augments(aug_method,image_name_with_path,patch_image);
							
						patch_image_label.append([patch_image,camera_model]);
						if save_data==True:
							imageName= image_name_with_path.split('\\')[-1][:-4];
							file_name = save_to_dir+'\\'+save_prefix+'_'+camera_model+'_'+imageName+'_Loc_X_'+str(X)+'_Y_'+str(Y)+'.'+save_format;
							image_name.append(file_name);
							cv2.imwrite(file_name,patch_image);
						
				
		for i in range(random_patch_left):
			camera_model = np.random.choice(list(image_dict.keys()))
			num_images = len(image_dict[camera_model]);
			index = np.random.randint(0,num_images,1)[0];
			image_name_with_path = image_dict[camera_model][index]
			
			img = cv2.imread(image_name_with_path);
			
			aug_method = np.random.choice(ImplementedDataAugMethods);
			aug_method_patchImage_flag = True;
			if aug_method  in [ 'JPEG_QF_70','JPEG_QF_90','RESIZE_BC_0p5','RESIZE_BC_0p8','RESIZE_BC_1p5','RESIZE_BC_2p0']:
				img = data_augments(aug_method,image_name_with_path)
				aug_method_patchImage_flag = False;
				
			H,W,C = img.shape;
			X = np.random.randint(p_w_half+1,W-p_w_half-1,1)[0];
			Y = np.random.randint(p_h_half+1,H-p_h_half-1,1)[0];
			patch_image= img[Y-p_h_half:Y+p_h_half,X-p_w_half:X+p_w_half,:]
			
			if aug_method_patchImage_flag:
				patch_image = data_augments(aug_method,image_name_with_path,patch_image);
			patch_image_label.append([patch_image,camera_model]);
			
			if save_data==True:
				imageName= image_name_with_path.split('\\')[-1][:-4];
				file_name = save_to_dir+'\\'+save_prefix+'_'+camera_model+'_'+imageName+'_Loc_X_'+str(X)+'_Y_'+str(Y)+'.'+save_format;
				cv2.imwrite(file_name,patch_image);
				image_name.append(file_name);
		#pdb.set_trace()
		#print (image_name);print('.............');
		patch_image_label = list(patch_image_label);
		assert(len(patch_image_label) == num_patch),"Issue with Data Generation Code"
		return patch_image_label
	
	
				
				
'''			
dataGenObg = DataGeneration();
for i, data in enumerate(dataGenObg.dataGeneratefromDirectory()):
	print(data);
	if( i ==3):
		break
'''
'''
print ('next');
print(next(dataGenObg.dataGeneratefromDirectory()))
print(next(dataGenObg.dataGeneratefromDirectory()))
print(next(dataGenObg.dataGeneratefromDirectory()))
print(next(dataGenObg.dataGeneratefromDirectory()))
'''
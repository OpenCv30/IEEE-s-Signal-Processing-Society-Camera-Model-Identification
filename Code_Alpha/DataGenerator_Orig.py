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

from DataAugUtils import *

ImplementedDataAugMethods = ['NoChange','Hflip','Vflip','R90','R270',
					   'JPEG_QF_70','JPEG_QF_90',
					   'RESIZE_BC_0p5','RESIZE_BC_0p8','RESIZE_BC_1p5','RESIZE_BC_2p0',
						'GammaCorr_0p8','GammaCorr_1p2'];

class DataGeneration():
	def __init__(self,
		trainDataPath=r'C:\Machinelearning\cameraModel\data\train',
		flip_H=True,
		flip_V=True,
		Rotation=True,
		JPEG_QF=True,
		RESIZE_BC=True,
		GAMMA_CORR=True,
		shear_range=0.0,
		Rescale_Intensity=True,
		rescale=1./255,
		data_format='channels_last'):
		
		self.trainDataPath = trainDataPath;
		self.flipH = flip_H;
		self.flipV = flip_V;
		self.rescale=rescale;
		self.Rotation  = Rotation;
		self.JPEG_QF_Compress=JPEG_QF;
		self.RESIZE_BC = RESIZE_BC;
		self.GammaCorr = GAMMA_CORR;
		self.shear_range = shear_range;
		self.Rescale_Intensity = Rescale_Intensity;
		self.data_format = data_format;
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
		batch_features = np.zeros((num_patch, patch_size[0], patch_size[1], n_chn),dtype=float) #bxhxwxc
		
		while True:
			batch_labels = [];
			data_label =self.patch_generation(patch_size,num_patch,num_class,self.dataDict_cameras_imgPaths,save_data,
							save_to_dir,save_prefix,save_format);
			for i in range(len(data_label)):
				batch_features[i]= np.float32(data_label[i][0])*self.rescale;
				batch_labels.append(data_label[i][1]);
			
			encoded_labels= self.LE.transform(list(batch_labels))
			cat_label_matrix = to_categorical(encoded_labels,num_classes=num_class);
			#pdb.set_trace()
			yield batch_features,cat_label_matrix
			
			
		
	
	def patch_generation(self,patch_size,num_patch,num_class,image_dict,
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
					img_org = cv2.imread(image_name_with_path);
					Horg,Worg,C = img_org.shape;
					
					#crop 10% along all dimension to get rid of boundary.
					H10 = int(Horg/10);W10 = int(Worg/10);
					img = img_org[H10:Horg-H10,W10:Worg-W10,:];
					H,W,C = img.shape;
					
					randomX = np.random.randint(p_w_half+1,W-p_w_half-1,num_patch_per_class_image);
					randomY = np.random.randint(p_h_half+1,H-p_h_half-1,num_patch_per_class_image);
					for X,Y in zip(randomX,randomY):
						patch_image= img[Y-p_h_half:Y+p_h_half,X-p_w_half:X+p_w_half,:]
						if patch_image is None:
							 pdb.set_trace()
						patch_image = self.random_transform(patch_image);
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
			img_org = cv2.imread(image_name_with_path);
			Horg,Worg,C = img_org.shape;
			#crop 10% along all dimension to get rid of boundary.
			H10 = int(Horg/10);W10 = int(Worg/10);
			img = img_org[H10:Horg-H10,W10:Worg-W10,:];
			H,W,C = img.shape;
			X = np.random.randint(p_w_half+1,W-p_w_half-1,1)[0];
			Y = np.random.randint(p_h_half+1,H-p_h_half-1,1)[0];
			
			patch_image= img[Y-p_h_half:Y+p_h_half,X-p_w_half:X+p_w_half,:]
			if patch_image is None:
				pdb.set_trace()
			patch_image = self.random_transform(patch_image);
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
	
				
	def random_transform(self, x):
		#print('Random Tran');#pdb.set_trace()
		if self.data_format == 'channels_last': 
			image_row_h_axis =0;
			image_col_w_axis =1;
			image_chn_axis =2;
		else:
			image_row_h_axis =1;
			image_col_w_axis =2;
			image_chn_axis =0;
		
		jpeg_comp_flag = False;
		if self.flipH:
			if np.random.random() < 0.5:
				x = flip_axis(x, image_col_w_axis);
				
		if self.flipV:
			if np.random.random() < 0.5:
				x = flip_axis(x, image_row_h_axis)

		if self.Rotation:
			if np.random.random() < 0.5:
				if np.random.random() < 0.5:
					x= Rotate(x,'R90');
				else:
					x= Rotate(x,'R270');

		if self.JPEG_QF_Compress:
			if np.random.random() < 0.5:
				jpeg_comp_flag =True;
				if np.random.random() < 0.5:
					x = JpegCompression_withImage(x,70)
				else:
					x = JpegCompression_withImage(x,90)
							
		if (self.RESIZE_BC) and (not jpeg_comp_flag):
			if np.random.random() < 0.5:
				factor = np.random.choice([0.5, 0.8, 1.5, 2]);
				x= Resize_Bicubic(x,factor)
				
		if self.GammaCorr:
			if np.random.random() < 0.5:
				if np.any(x < 0):
					pdb.set_trace()
				if np.random.random() < 0.5:
					x = GammaCorr(x,0.8)
				else:
					x = GammaCorr(x,1.2);
		
		if self.Rescale_Intensity:
			if np.random.random() <0.5:
				x = rescale_intesity(x);
		
		if self.shear_range:
			shear = np.random.uniform(-self.shear_range, self.shear_range)
		else:
			shear = 0
		
		if shear != 0:
			shear_matrix = np.array([[1, -np.sin(shear), 0],
									[0, np.cos(shear), 0],
									[0, 0, 1]])
			transform_matrix = shear_matrix ;
		
			h, w = x.shape[image_row_h_axis], x.shape[image_col_w_axis]
			transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
			x = apply_transform(x, transform_matrix, image_chn_axis)	
		
		return x;
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
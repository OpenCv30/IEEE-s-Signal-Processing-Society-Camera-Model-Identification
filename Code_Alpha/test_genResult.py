import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import pdb
import os
import numpy as np
from threading import Thread
from multiprocessing import Process
from time import sleep
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import glob
import pandas as pd
from model import ResNetModel
from DataGenerator_Orig import DataGeneration

def mid_intensity_high_texture(img):
	"""
	:param img: 2D or 3D ndarray. Values are expected in [0,1] if img is float, in [0,255] if img is uint8
	:return score: score in [0,1]. Score tends to 1 as intensity is not saturated and high texture occurs
	https://bitbucket.org/polimi-ispl/camera-model-identification-with-cnn/src/d9cceca1c7501e0866cc4c6b9e4c620700943cb2/patch_extractor.py?at=master&fileviewer=file-view-default
	"""
	if img.dtype == np.uint8:
		img = img / 255.

	mean_std_weight = .7
	num_ch = 1 if img.ndim == 2 else img.shape[-1]
	img_flat = img.reshape(-1, num_ch)
	ch_mean = img_flat.mean(axis=0)
	ch_std = img_flat.std(axis=0)
	ch_mean_score = -4 * ch_mean ** 2 + 4 * ch_mean
	ch_std_score = 1 - np.exp(-2 * np.log(10) * ch_std)
	ch_mean_score_aggr = ch_mean_score.mean();ch_std_score_aggr = ch_std_score.mean()
	score = mean_std_weight * ch_mean_score_aggr + (1 - mean_std_weight) * ch_std_score_aggr
	return score
		
def generate_cameraClass(thdName,imgList,trained_model_obj,patch_size_hxw,clss_encoder,scale,batch_size,outPut_dict):
	print('Running:-ThdName :{}'.format(thdName));
	threshold_to_decide_patch = 0.75;
	for index,file in enumerate(imgList):
		imgOrg = cv2.imread(file);
		Horg,Worg,C = imgOrg.shape;
		assert(C==3),"Need RGB Image to run"
		#take out 10 % from boundary-
		if 0:
			H10= int(Horg/10); W10 = int(Worg/10);
			imgProcess = imgOrg[H10:Horg-H10,W10:Worg-W10,:];
		else:
			imgProcess = imgOrg
		H,W,C = imgProcess.shape;
		ph,pw = patch_size_hxw;
		ph_half = int(ph/2); 
		pw_half= int(pw/2);
		Hloc,Wloc = np.mgrid[ph_half:H:ph_half,pw_half:W:pw_half] #index will come till W-pw_half
		hwLoc = np.column_stack((Hloc.ravel(),Wloc.ravel()))
		patch_imageList=[];scoreList=[];
		'''
		people = numpy.array(people)
		ages = numpy.array(ages)
		inds = ages.argsort()
		sortedPeople = people[inds]
		or
		[x for y, x in sorted(zip(Y, X))]
		'''
		for h,w in hwLoc:
			patch = imgProcess[h-ph_half:h+ph_half,w-pw_half:w+pw_half,:]
			score = mid_intensity_high_texture(patch);
			patch = patch*scale;
			patch_imageList.append(patch);
			scoreList.append(score);
		
		'''Filter out score less than threshold '''
		sorted_patch_images = [x  for y,x in sorted(zip(scoreList,patch_imageList)) if (y > threshold_to_decide_patch)]
		if len(sorted_patch_images) < 10:
			 sorted_patch_images = [x  for y,x in sorted(zip(scoreList,patch_imageList))] # take all
		#pdb.set_trace()
		patch_image=np.array(sorted_patch_images);
		y_predicted=trained_model_obj.Model.predict(patch_image, batch_size=batch_size, verbose=2)
		n,classes= y_predicted.shape;
		class_histogram = np.zeros([classes,1],np.float32);
		for prediction in y_predicted:
			index = int(np.argmax(prediction))
			class_histogram[index]+=1;
		class_index = np.argmax(class_histogram);
		camera_model= clss_encoder.inverse_transform(class_index);
		
		image_image = file.split('\\')[-1]
		outPut_dict['fname'].append(image_image);
		outPut_dict['camera'].append(camera_model);
		

''' Model Params'''
patch_width= 64;
patch_height =64;
img_channel =3;
batch_size=32;
patch_size_hxw =(patch_height,patch_width);
scale = 1.0/255;
IS_THREADED = False;
modelPath =r'C:\Machinelearning\cameraModel\models\weights_sgd__Loaded_Weight_29_12_2017_. 11-1.952.hdf5'
complete_model_with_weight = r'C:\Machinelearning\cameraModel\approch2\finalModel\epoch_Model\Resnet50_FineTuning_weights_sdg__2_01_2018_.000-1.513.hdf5';
input_image_shape= (patch_height,patch_width,img_channel);
MODEL_ONLY_WEIGHT = False

if not IS_THREADED and MODEL_ONLY_WEIGHT :
	''' Load Model'''
	resnet_object = ResNetModel(input_image_shape);
	resnet_object.Model.load_weights(modelPath); #load_model('path_to_model);
	print('Model Loded!');
	
if not MODEL_ONLY_WEIGHT:
	resnet_object = load_model(complete_model_with_weight);
	print("Full Model is loaded weight");

''' read Images '''
testImgFolderPath = r'C:\Machinelearning\cameraModel\data\test'
imageFiles = glob.glob(testImgFolderPath+'\\*.tif');
#imageFiles = imageFiles[0:2];
mydata = {'fname':[],'camera':[]};

'''to read class-as we have used Encoder() -Need to get rid of this'''
trainDataFolderPath = r'C:\Machinelearning\cameraModel\data\train'
test_datagen = DataGeneration(trainDataPath=trainDataFolderPath);
clss_encoder = test_datagen.LE;
if IS_THREADED:
	''' threading call'''
	dataLen = len(imageFiles);
	nThread =1;
	partionindex = int(dataLen/4);
	threadList=[];
	outputdictThd=[];
	#for thread call create 4 model objects
	modelTrainedList=[];
	for i in range(nThread):
		resnet_object = ResNetModel(input_image_shape);
		resnet_object.Model.load_weights(modelPath);
		modelTrainedList.append(resnet_object);
	print('Loaded all model,Number = {}'.format(len(modelTrainedList)))
	#pdb.set_trace()
	for i in range(nThread):
		thdName= 'Thd_Name_'+str(i);
		if ( i== nThread-1):
			lastIndex = len(imageFiles);
		else:
			lastIndex = (i+1)*partionindex;	
		imgList = imageFiles[i*partionindex:lastIndex];
		outPut_dict = {'fname':[],'camera':[]};
		thd = Process(target=generate_cameraClass,args=(thdName,imgList,modelTrainedList[i],patch_size_hxw,clss_encoder,scale,batch_size,outPut_dict)); #Thread 
		threadList.append(thd);
		outputdictThd.append(outPut_dict);
		
	for thd	 in threadList:
		thd.start();
		
	for thd	 in threadList:
		thd.join();
	for out in outputdictThd:
		print(out);
	pdb.set_trace()

else:
	threshold_to_decide_patch =0.75
	for indeximg,file in enumerate(imageFiles):
		imgOrg = cv2.imread(file);
		Horg,Worg,C = imgOrg.shape;
		assert(C==3),"Need RGB Image to run"
		#take out 10 % from boundary-
		if 0:
			H10= int(Horg/10); W10 = int(Worg/10);
			imgProcess = imgOrg[H10:Horg-H10,W10:Worg-W10,:];
		else:
			imgProcess = imgOrg
		H,W,C = imgProcess.shape;
		ph,pw = patch_size_hxw;
		ph_half = int(ph/2); 
		pw_half= int(pw/2);
		Hloc,Wloc = np.mgrid[ph_half:H:ph_half,pw_half:W:pw_half] #index will come till W-pw_half
		hwLoc = np.column_stack((Hloc.ravel(),Wloc.ravel()))
		patch_imageList=[];scoreList=[];
		'''
		people = numpy.array(people)
		ages = numpy.array(ages)
		inds = ages.argsort()
		sortedPeople = people[inds]
		or
		[x for y, x in sorted(zip(Y, X))]
		'''
		for h,w in hwLoc:
			patch = imgProcess[h-ph_half:h+ph_half,w-pw_half:w+pw_half,:]
			score = mid_intensity_high_texture(patch);
			patch = np.float32(patch)*scale;
			patch_imageList.append(patch);
			scoreList.append(float(score));
		#pdb.set_trace()
		'''Filter out score less than threshold '''
		'''#sorted_patch_images = [print(y,x)	for y,x in sorted(zip(scoreList,patch_imageList)) if (y > threshold_to_decide_patch)]
		for y,x in sorted(zip(scoreList,patch_imageList)):
			if (y > threshold_to_decide_patch):
				print (x,y)
				sorted_patch_images.append(x);
		if len(sorted_patch_images) < 10:
			 sorted_patch_images = [x  for y,x in sorted(zip(scoreList,patch_imageList))] # take all'''
			 
		
		sorted_patch_images=[];
		for patch,score in zip(patch_imageList,scoreList):
			if (score > threshold_to_decide_patch):
				sorted_patch_images.append(patch);
				
		#pdb.set_trace()
		if len(sorted_patch_images) < 10:
			sorted_patch_images = patch_imageList;
		
		patch_image=np.array(sorted_patch_images);
		y_predicted=resnet_object.Model.predict(patch_image, batch_size=batch_size, verbose=1)
		n,classes= y_predicted.shape;
		class_histogram = np.zeros([classes,1],np.float32);
		for prediction in y_predicted:
			index = int(np.argmax(prediction))
			class_histogram[index]+=1;
		class_index = np.argmax(class_histogram);
		camera_model= test_datagen.LE.inverse_transform(class_index);
		
		image_image = file.split('\\')[-1]
		mydata['fname'].append(image_image);
		mydata['camera'].append(camera_model);
		if ( (indeximg % 100) ==0 ):
			print('Image Done_',' ',indeximg);


	#print(mydata)
df = pd.DataFrame(mydata, columns = ['fname', 'camera'])
df.to_csv('./test_result_2_01_2018.csv', encoding='utf-8', index=False)

'''
(Pdb) binary_vector.T
array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
(Pdb) test_datagen.LE.inverse_transform(binary_vector.T)
array([['HTC-1-M7', 'HTC-1-M7', 'HTC-1-M7', 'LG-Nexus-5x', 'HTC-1-M7',
		'HTC-1-M7', 'HTC-1-M7', 'HTC-1-M7', 'HTC-1-M7', 'HTC-1-M7']],
	  dtype='<U20')
(Pdb) test_datagen.classLabel
['Motorola-Nexus-6', 'iPhone-4s', 'iPhone-6', 'Samsung-Galaxy-Note3', 'Motorola-
X', 'HTC-1-M7', 'LG-Nexus-5x', 'Motorola-Droid-Maxx', 'Sony-NEX-7', 'Samsung-Gal
axy-S4']
(Pdb) test_datagen.LE.inverse_transform([1])
array(['LG-Nexus-5x'],
	  dtype='<U20')
(Pdb) test_datagen.LE.inverse_transform([3])
array(['Motorola-Nexus-6'],
	  dtype='<U20')
(Pdb) test_datagen.LE.inverse_transform([0])
array(['HTC-1-M7'],
	  dtype='<U20')
(Pdb) test_datagen.LE.inverse_transform([1])
array(['LG-Nexus-5x'],
	  dtype='<U20')
(Pdb) test_datagen.LE.transform(test_datagen.classLabel)
array([3, 8, 9, 5, 4, 0, 1, 2, 7, 6], dtype=int64)
(Pdb) test_datagen.LE.inverse_transform([3])
array(['Motorola-Nexus-6'],
	  dtype='<U20')
'''	  
	
		
	
	
	


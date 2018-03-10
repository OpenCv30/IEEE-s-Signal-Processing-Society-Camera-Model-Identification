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
from skimage.exposure import adjust_gamma,rescale_intensity
from skimage.io import imsave 
import scipy as sp


'''
Possible Transform

JPEG compression with quality factor = 70
JPEG compression with quality factor = 90
resizing (via bicubic interpolation) by a factor of 0.5
resizing (via bicubic interpolation) by a factor of 0.8
resizing (via bicubic interpolation) by a factor of 1.5
resizing (via bicubic interpolation) by a factor of 2.0
gamma correction using gamma = 0.8
gamma correction using gamma = 1.2
'''

def Hflip(img):
	return np.fliplr(img);
	
def Vflip(img):
	return np.flipud(img);
	
def flip_axis(x, axis):
	x = np.asarray(x).swapaxes(axis, 0)
	x = x[::-1, ...]
	x = x.swapaxes(0, axis)
	return x

def Rotate(img,degree):
	if(degree=='R90'):
		return np.rot90(img);
	elif(degree =='R270'):
		return np.rot90(np.rot90(np.rot90(img)));
	else:
		assert(False),"Rotation Method ={} is not implemented.".format(degree)
		
def GammaCorr(img,gamma):
	if( gamma == 0.8):
		return adjust_gamma(img,0.8); #out put will have same dtype as input	
	elif(gamma ==1.2):
		return adjust_gamma(img,1.2);	
	else:
		assert(False),"Gamma Method ={} is not implemented.".format(gamma)

def JpegCompression_withImage(img,jpegQf):
	cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	imgPIL = Image.fromarray(cv2_im)
	imsave('./dummy_jpegCompression.jpg',imgPIL,quality=jpegQf) #[1-100]
	img=cv2.imread('./dummy_jpegCompression.jpg');
	return img
	
def JpegCompression(img_full_path,jpegQf):
	#we will first read using PIL.Image, and save it with known JPEG factor
	imgPIL = Image.open(img_full_path);
	imsave('./dummy_jpegCompression.jpg',imgPIL,quality=jpegQf) #[1-100]
	img=cv2.imread('./dummy_jpegCompression.jpg');
	return img
	
def JpegCompression_withImage_valid(img,jpegQf):
	cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	imgPIL = Image.fromarray(cv2_im)
	imsave('./dummy_jpegCompression_valid.jpg',imgPIL,quality=jpegQf) #[1-100]
	img=cv2.imread('./dummy_jpegCompression_valid.jpg');
	return img
	
def Resize_Bicubic(img,factor):
	#if factor <1 then first up sample and the down-sample to same size and vice-versa
	H,W,C = img.shape;
	if( factor < 1):
		factorUp = 1/factor;
		imgUP=sp.misc.imresize(img,factorUp,interp='bicubic');
		imgOrigSize=sp.misc.imresize(imgUP,(H,W,C),interp='bicubic');
		
	else:
		factorDwn = 1/factor;
		imgDwn=sp.misc.imresize(img,factorDwn,interp='bicubic');
		imgOrigSize=sp.misc.imresize(imgDwn,(H,W,C),interp='bicubic');
	return imgOrigSize;

def transform_matrix_offset_center(matrix, x, y):
	o_x = float(x) / 2 + 0.5
	o_y = float(y) / 2 + 0.5
	offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
	reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
	transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
	return transform_matrix

def apply_transform(x,
					transform_matrix,
					channel_axis=0,
					fill_mode='nearest',
					cval=0.):
	"""Apply the image transformation specified by a matrix.

	# Arguments
		x: 2D numpy array, single image.
		transform_matrix: Numpy array specifying the geometric transformation.
		channel_axis: Index of axis for channels in the input tensor.
		fill_mode: Points outside the boundaries of the input
			are filled according to the given mode
			(one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
		cval: Value used for points outside the boundaries
			of the input if `mode='constant'`.

	# Returns
		The transformed version of the input.
	"""
	x = np.rollaxis(x, channel_axis, 0)
	final_affine_matrix = transform_matrix[:2, :2]
	final_offset = transform_matrix[:2, 2]
	channel_images = [ndi.interpolation.affine_transform(
		x_channel,
		final_affine_matrix,
		final_offset,
		order=0,
		mode=fill_mode,
		cval=cval) for x_channel in x]
	x = np.stack(channel_images, axis=0)
	x = np.rollaxis(x, 0, channel_axis + 1)
	return x

def rescale_intesity(img):
	p2, p98 = np.percentile(img, (2, 98))
	img_rescale = rescale_intensity(img, in_range=(p2, p98))
	return img_rescale;
'''		
def image_transform(imgData,method,batch_size,patch_size_chn):
	transform_data= np.zeros(batch_size,patch_size_chn); #patch_size_chn=(h,w,c)
	for i in range(batch_size):
		method= np.random.choice(ImplementedDataAugMethods);
'''		
def data_augments(method,image_name_path=None,patchImage=None):
	'''For few method only patch image is sufficient but for other full image is required '''
	
	assert(not((image_name_path==None)and(patchImage==None))),"Both can't be None- at data_augments"
	
	img= np.float32(cv2.imread(image_name_path));
	patchImage = np.float32(patchImage)
	if(method =='NoChange'):
		return patchImage;
	elif(method == 'Hflip'):
		return Hflip(patchImage);
	elif(method == 'Vflip'):
		return Vflip(patchImage);
	elif(method == 'R90'):
		return Rotate(patchImage,'R90');
	elif(method == 'R270'):
		return Rotate(patchImage,'R270');
	elif(method == 'RESIZE_BC_0p5'):
		return Resize_Bicubic(img,0.5);
	elif(method == 'RESIZE_BC_0p8'):
		return Resize_Bicubic(img,0.8);
	elif(method == 'RESIZE_BC_1p5'):
		return Resize_Bicubic(img,1.5);
	elif(method == 'RESIZE_BC_2p0'):
		return Resize_Bicubic(img,2.0);
	elif(method == 'JPEG_QF_70'):
		return JpegCompression(image_name_path,70);
	elif(method == 'JPEG_QF_90'):
		return JpegCompression(image_name_path,90);
	elif(method == 'GammaCorr_0p8'):
		return GammaCorr(patchImage,0.8);
	elif(method == 'GammaCorr_1p2'):
		return GammaCorr(patchImage,1.2);
	else:
		assert(False),"Given Data Aug method= {} is not implemented".format(method)
					

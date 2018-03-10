import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt


class DataAugmentation():
	def __init__(self,patch_size,batch_size,image_dictionary_camera_img):
		self.patch_size=patch_size;
		self.num_image = batch_size;
		self.method='UNIFORM_ALL_CLASS';
		self.dict_Camera_img =image_dictionary_camera_img;
	
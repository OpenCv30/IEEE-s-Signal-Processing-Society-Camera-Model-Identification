import model_online
import numpy as np
from keras.utils import plot_model

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 224, 224
# The CIFAR10 images are RGB.
img_channels = 3

resnet_model = model_online.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
plot_model(resnet_model,show_shapes =True, to_file='./model_resnet18_online.png');
# -*-coding:utf8-*-
from read_data import read_file
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import random
import numpy as np

img_size = 224
MEAN_PIXEL = np.asarray((123.68, 116.779, 103.939), np.float32)


def data_handle(path):
	imgs, labels, counter = read_file(path)
	imgs -= MEAN_PIXEL
	X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2,
	                                                    random_state=random.randint(0, 100))

	X_train = X_train.reshape(X_train.shape[0], img_size, img_size, 3) / 255.0
	X_test = X_test.reshape(X_test.shape[0], img_size, img_size, 3) / 255.0

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	# 多目标的label:one hot coding
	Y_train = np_utils.to_categorical(y_train)
	Y_test = np_utils.to_categorical(y_test)

	num_classes = counter
	print('handle successful')
	return X_train, X_test, Y_train, Y_test, num_classes


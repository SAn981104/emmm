# -*-coding:utf8-*-
import os
from keras.preprocessing import image
import numpy as np

IMG_SIZE = 224


def read_file(path):
	image_list = []
	label_list = []
	dir_counter = 0

	for dir in os.listdir(path):
		lower_path = os.path.join(path, dir)

		for image_ in os.listdir(lower_path):
			if image_.endswith('jpg'):
				img = image.load_img(os.path.join(lower_path, image_), target_size=(IMG_SIZE, IMG_SIZE))
				img = image.img_to_array(img)
				image_list.append(img)
				label_list.append(dir_counter)

		dir_counter += 1

	image_list = np.array(image_list, np.float32)
	# image_list = [image_list]
	print('Read data successful')
	return image_list, label_list, dir_counter


def read_name_list(path):
	name_list = []
	for child_dir in os.listdir(path):
		name_list.append(child_dir)
	return name_list



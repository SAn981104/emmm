import os
import shutil

tar_path = 'data'


def make_dir(filename):
	name = filename.split('_')[0]
	new_p = tar_path + '\\' + name

	if not os.path.exists(new_p):
		os.makedirs(new_p)
	return new_p


def move_file(file, new_path):
	shutil.move(file, new_path)


for file in os.listdir(tar_path):
	if file.endswith('.jpg'):
		file_path = tar_path + '\\' + file
		move_file(file_path, make_dir(file))

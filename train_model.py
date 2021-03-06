# -*-coding:utf-8*-
from data_handle import data_handle
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import PReLU
# from keras.preprocessing import image
import numpy as np
from keras import optimizers

drop = 0.3
FILE_PATH = "model\\model.h5"
IMAGE_SIZE = 224
batch_size = 68


def forward():
	model = Sequential()
	model.add(Conv2D(filters=64, kernel_size=3, activation='relu',
	                 padding='same', data_format='channels_first', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

	model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', data_format='channels_first'))
	model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

	model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
	model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

	model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
	model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
	# model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))

	model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
	# model.add(Conv2D(filters=256, kernel_size=3, activation='linear', padding='same'))
	# model.add(PReLU())
	model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

	model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
	model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
	# model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))

	model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
	# model.add(Conv2D(filters=512, kernel_size=3, activation='linear', padding='same'))
	# model.add(PReLU())
	model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

	model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
	model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
	# model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))

	model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
	# model.add(Conv2D(filters=512, kernel_size=3, activation='linear', padding='same'))
	# model.add(PReLU())
	model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(drop))

	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(drop))

	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
	model.summary()

	return model


def backward(model):
	adam = optimizers.Adam(lr=0.01, decay=0.1)
	# 下面这个loss就是个弟弟
	# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
	# 这个还行
	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	model.fit(X_train, Y_train, epochs=5, batch_size=batch_size)
	# model.fit_generator(generate_batch_size(X_train, Y_train, batch_size, num_classes),
	#                     steps_per_epoch=(num_classes // batch_size), epochs=15, workers=2)


def evaluate_model(model):
	print('\nTesting---------------')
	loss, accuracy = model.evaluate(X_test, Y_test)

	print('test loss:', loss)
	print('test accuracy:', accuracy * 100)


def save(model, file_path=FILE_PATH):
	print('Saving successfully.')
	model.save(file_path)


def load(file_path=FILE_PATH):
	print('Loading successfully.')
	model = load_model(file_path)

	return model


def predict(model, img):
	img = img.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3)).astype('float32')
	img = img / 255.0

	result = model.predict_proba(img)
	max_index = np.argmax(result)

	return max_index, result[0][max_index]


if __name__ == '__main__':
	X_train, X_test, Y_train, Y_test, num_classes = data_handle('A:\\python_work\\data')

	m = forward()
	backward(m)
	evaluate_model(m)
	save(m)

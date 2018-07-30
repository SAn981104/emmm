# -*-coding:utf-8*-
from data_handle import data_handle
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
# from keras.preprocessing import image
import numpy as np
from keras import optimizers

drop = 0.5

FILE_PATH = "model\\model.h5"
IMAGE_SIZE = 224


def forward():
	model = Sequential()
	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
	                 padding='same', dim_ordering='th',
	                 input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

	model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', dim_ordering='th'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

	model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

	model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

	model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

	model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

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
	adam = optimizers.Adam(lr=0.01)
	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	model.fit(X_train, Y_train, epochs=15, batch_size=32)


def evaluate_model(model):
	print('\nTesting---------------')
	loss, accuracy = model.evaluate(X_test, Y_test)

	print('test loss:', loss)
	print('test accuracy:', accuracy)


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

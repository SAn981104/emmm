# coding: utf-8
import cv2
import dlib
import os
import random

name = input()
output_dir = 'A:\\python_work\\work\\sb' + name
size = 224

if not os.path.exists(output_dir):
	os.makedirs(output_dir)


def relight(image, light=1, bias=0):
	w = image.shape[1]
	h = image.shape[0]

	for x in range(0, w):
		for j in range(0, h):
			for c in range(3):
				tmp = int(image[j, i, c] * light + bias)
				if tmp > 255:
					tmp = 255
				elif tmp < 0:
					tmp = 0
				image[j, x, c] = tmp
	return image


detector = dlib.get_frontal_face_detector()
camera = cv2.VideoCapture(0)

index = 1

while index <= 10:
	print(' picture %s' % index)
	success, img = camera.read()
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = detector(gray_img, 1)

	for i, d in enumerate(faces):
		x1 = d.top() if d.top() > 0 else 0
		y1 = d.bottom() if d.bottom() > 0 else 0
		x2 = d.left() if d.left() > 0 else 0
		y2 = d.right() if d.right() > 0 else 0

		face = img[x1:y1, x2:y2]

		face = relight(face, random.uniform(0.5, 1.5), random.randint(-35, 35))
		face = cv2.resize(face, (size, size))
		cv2.imshow('image', face)
		cv2.imwrite(output_dir + '\\' + str(index) + '.jpg', face)

		index += 1
	key = cv2.waitKey(30) & 0xff
	if key == 27:
		break
	else:
		print('Finish')
		break

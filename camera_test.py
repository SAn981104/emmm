# -*- coding:utf-8 -*-

import cv2
from read_data import read_name_list
import dlib
from keras.models import load_model
from train_model import predict


def main():
	detector = dlib.get_frontal_face_detector()

	cam = cv2.VideoCapture(0)
	success, frame = cam.read()

	while success and cv2.waitKey(1) == -1:
		success, frame = cam.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray, 1)

		for i, d in enumerate(faces):
			x = d.top() if d.top() > 0 else 0
			w = d.bottom() if d.bottom() > 0 else 0
			y = d.left() if d.left() > 0 else 0
			h = d.right() if d.right() > 0 else 0

			det = frame[x:x + w, y:y + h]
			det = cv2.resize(det, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

			label, prob = predict(model, det)
			if prob > 0.9:
				show_name = name_list[label]
			else:
				show_name = 'Unknown'
			cv2.putText(frame, show_name, (y, w - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
			frame = cv2.rectangle(frame, (y, x), (h, w), (255, 0, 0), 2)
		cv2.imshow("Camera", frame)

	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	img_size = 224
	name_list = read_name_list('A:\\python_work\\data')
	# num_classes = len(name_list)

	model = load_model('model\\model.h5')

	main()

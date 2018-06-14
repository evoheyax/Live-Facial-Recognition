from __future__ import print_function, division
import pca_svm_functions as psf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from facial_recognizer import facial_recognizer

import cv2

import numpy as np

if __name__ == "__main__":

	face_decection = facial_recognizer("haarcascade_frontalface_default.xml", "svm_training_photos", "svm_training_photos_cleaned")

	print("Opening Video Camera...")
	cap = cv2.VideoCapture(0)

	#fourcc = cv2.cv.CV_FOURCC(*'XVID')
	#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

	ret, frame = cap.read()

	if(cap.isOpened()):
		print("Video Camera Opened Successfully")
	else:
		print("Error Opening Video Camera")

	total = 0
	while(True):
		ret, frame = cap.read()

		#out.write(frame)

		people = face_decection.identify(frame)
		for person in people:
			print("{}: Found {}".format(total, person))
		total = total + 1

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	#out.release()
	cv2.destroyAllWindows()

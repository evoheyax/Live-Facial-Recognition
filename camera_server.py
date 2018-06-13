from __future__ import print_function, division
import pattern_recog_func as prf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

import numpy as np

if __name__ == "__main__":
	#names_dict = {2: "Luke", 0: "Gilbert", 1: "Janet"}

	# Load and Train SVM
	print("Loading images...")

	images, y, names_dict = prf.load_images("svm_training_photos", 4)

	images = np.array(images)

	y = np.array(y)

	print("Cropping and interpoolating images...")

	for i in range(len(images)):
		images[i] = prf.crop_and_interpool_image(images[i])

	X = np.vstack(images)

    # Open Video Capture
	cap = cv2.VideoCapture(0)

	if(cap.isOpened()):
		print("Video Camera 1 Opened Successfully")
	else:
		print("error opening camera stream")

	total = 0
	while(True):
		ret, frame = cap.read()

		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#plt.imshow(gray)
		#plt.show()

		people = prf.identify(X, y, names_dict, frame)
		for person in people:
			print("{}: Found {}".format(total, person))
		total = total + 1

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

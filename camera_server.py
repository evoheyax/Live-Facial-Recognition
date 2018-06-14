from __future__ import print_function, division
import pattern_recog_func as prf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

import numpy as np

if __name__ == "__main__":
	print("Loading images...")
	images, y, names_dict = prf.load_images()
	images = np.array(images)

	y = np.array(y)
	X = np.vstack(images)
    
	print("Training with PCA+SVM...")
	md_pca, X_proj = prf.pca_X(X, n_comp = 50)           
	md_clf = prf.svm_train(X_proj, y)

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

		#plt.imshow(frame)
		#plt.show()

		#out.write(frame)

		people = prf.identify(md_pca, md_clf, X_proj, names_dict, frame)
		for person in people:
			print("{}: Found {}".format(total, person))
		total = total + 1

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	#out.release()
	cv2.destroyAllWindows()

from __future__ import print_function, division
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

from sklearn.decomposition import PCA
from sklearn import svm

from scipy.interpolate import interp2d, RectBivariateSpline

import os

from itertools import izip

def interpol_im(im, dim1 = 8, dim2 = 8, plot_new_im = False, cmap = 'binary', grid_off = False):
    
    if((len(im.shape)) == 3):
        im = im[:, :, 0]
    
    x = np.arange(im.shape[1])
    y = np.arange(im.shape[0])
    
    f2d = interp2d(x, y, im)
    
    x_new = np.linspace(0, im.shape[1], dim1)
    y_new = np.linspace(0, im.shape[0], dim2)
    
    new_im = f2d(x_new, y_new)
    if(plot_new_im):
        plt.grid(True)
        if(grid_off):
            plt.grid(False)
        plt.imshow(new_im, cmap = cmap)
        plt.show()
    
    new_im_flat = new_im.flatten()
    return new_im, new_im_flat

def pca_svm_pred(imfile, md_pca, md_clf, dim1 = 45, dim2 = 60):
    
    intp_img, intp_img_flat = interpol_im(imfile, dim1 = dim1, dim2 = dim2, plot_new_im = False)
    
    intp_img_flat = intp_img_flat.reshape(1, -1)
    
    X_proj = md_pca.transform(intp_img_flat)

    return md_clf.predict(X_proj)

def pca_X(X, n_comp = 10):
    md_pca = PCA(n_comp, whiten = True)
    
    # finding pca axes
    md_pca.fit(X)
    
    # projecting training data onto pca axes
    X_proj = md_pca.transform(X)
    
    return md_pca, X_proj

def rescale_pixel(unseen, ind = 0):
    unseen_rescaled = np.array((unseen * 15), dtype=np.int)
    for i in range(8):
        for j in range(8):
            if(unseen_rescaled[i, j] == 0):
                unseen_rescaled[i, j] = 15
            else:
                unseen_rescaled[i, j] = 0

    return unseen_rescaled

def svm_train(X, y, gamma = 0.001, C = 100):
    md_clf = svm.SVC(kernel='rbf', class_weight='balanced', gamma=gamma, C=C)
    
    # apply SVM to training data and draw boundaries.
    md_clf.fit(X, y)
    
    return md_clf

def crop_and_interpool_image(image):
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
                                     gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(30, 30),
                                     flags = cv2.CASCADE_SCALE_IMAGE
                                     )

    for (x, y, w, h) in faces:
        image = image[y:y+h, x:x+w]
    image, image_flattened = interpol_im(image, 45, 60)
    return image_flattened

def load_images():
	main_folder = "svm_training_photos"
	clean_folder = "svm_training_photos_cleaned"
	images = []
	targets = []
	count = 0

	people_list_main = os.listdir(main_folder)
	people_list_clean = os.listdir(clean_folder)

	people = dict()

	count = 0
	while count < len(people_list_main):
		person = people_list_main[count]
        people[count] = person
        count = count + 1

        if(person in people_list_clean):
			os.mkdir(people_list_clean+person)

			print("Cropping and interpoolating images...")

            spec_folder = main_folder+"/"+person
            print("Spec Folder: ", spec_folder)

        	for filename in os.listdir(spec_folder):
            	img = cv2.imread(os.path.join(spec_folder, filename))
            	if img is not None:
                	img = crop_and_interpool_image(img)
					cv2.imwrite(people_list_clean+person+filename,img)

    print("in first loop")
    for person in people_list:
        spec_folder = clean_folder+"/"+person
        print("Spec Folder: ", spec_folder)
        for filename in os.listdir(spec_folder):
            img = cv2.imread(os.path.join(spec_folder, filename))
            if img is not None:
                targets.append(people_list.index(person))
                images.append(img)

    return (images, targets, people)

def identify(md_pca, md_clf, X_proj, names_dict, frame):
	cascPath = "haarcascade_frontalface_default.xml"
	faceCascade = cv2.CascadeClassifier(cascPath)
  
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
                                         frame_gray,
                                         scaleFactor=1.3,
                                         minNeighbors=5,
                                         minSize=(50, 50),
                                         flags = cv2.CASCADE_SCALE_IMAGE
                                         )
	people = list()
	for (x, y, w, h) in faces:
		im = frame[y:y+h, x:x+w]
		prediction = pca_svm_pred(im, md_pca, md_clf)[0]
		people.append(names_dict[prediction])

	return people

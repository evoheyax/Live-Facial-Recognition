from __future__ import print_function, division
import numpy as np
import pca_svm_functions as psf
import cv2
import os

class facial_recognizer:
    def __init__(self, cascPath, main_folder, clean_folder):
        self.faceCascade = cv2.CascadeClassifier(cascPath)
        self.main_folder = main_folder
        self.clean_folder = clean_folder

        print("Loading images...")
        X, y, self.people_dict = self.load_image_data()
    
        print("Training with PCA+SVM...")
        self.md_pca, self.X_proj = psf.pca_X(X, n_comp = 50)           
        self.md_clf = psf.svm_train(self.X_proj, y)

    def crop_facial_image(self, image, cascPath):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            image = image[y:y+h, x:x+w]

        return image

    def clean_images(self, people_list_main, people_list_clean):
        count = 0
        while count < len(people_list_main):
            person = people_list_main[count]
            print("Checking images for: "+person)
            count = count + 1

            if(person not in people_list_clean):
                os.mkdir(self.clean_folder+"/"+person)

                print("    Cropping images")

                spec_folder = self.main_folder+"/"+person

                for filename in os.listdir(spec_folder):
                    img = cv2.imread(os.path.join(spec_folder, filename))
                    if img is not None:
                        new_img = self.crop_facial_image(img)

                        cv2.imwrite(self.clean_folder+"/"+person+"/"+filename, new_img)

    def load_images_cleaned(self, people_list_clean):
        images = []
        targets = []

        for person in people_list_clean:
            spec_folder = self.clean_folder+"/"+person

            for filename in os.listdir(spec_folder):
                img = cv2.imread(os.path.join(spec_folder, filename))

                if img is not None:
                    interpulated_img = psf.interpol_im(img, 45, 60)

                    targets.append(people_list_clean.index(person))
                    images.append(interpulated_img)

        return images, targets

    def load_image_data(self):

        people_list_main = os.listdir(self.main_folder)
        people_list_clean = os.listdir(self.clean_folder)

        people = dict()
        count = 0
        for person in people_list_main:
            people[count] = person
            count = count + 1

        self.clean_images(people_list_main, people_list_clean)
        images, targets = self.load_images_cleaned(people_list_main)

        images = np.array(images)
        X = np.vstack(images)
        y = np.array(targets)

        return (X, y, people)

    def identify(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50), flags = cv2.CASCADE_SCALE_IMAGE)
        people = list()
        for (x, y, w, h) in faces:
            im = frame[y:y+h, x:x+w]
            prediction = psf.pca_svm_pred(im, self.md_pca, self.md_clf)[0]
            people.append(self.people_dict[prediction])

        return people

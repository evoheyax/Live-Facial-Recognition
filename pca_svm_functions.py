from __future__ import print_function, division
import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from scipy.interpolate import interp2d
import cv2

def pca_svm_pred(imfile, md_pca, md_clf, dim1 = 45, dim2 = 60):
    intp_img = interpol_im(imfile, dim1 = dim1, dim2 = dim2)
    intp_img_flat = intp_img.reshape(1, -1)
    X_proj = md_pca.transform(intp_img_flat)

    return md_clf.predict(X_proj)

def pca_X(X, n_comp = 10):
    md_pca = PCA(n_comp, whiten = True)
    md_pca.fit(X) # finding pca axes
    X_proj = md_pca.transform(X) # projecting training data onto pca axes
    
    return md_pca, X_proj

def svm_train(X, y, gamma = 0.001, C = 100):
    md_clf = svm.SVC(kernel='rbf', class_weight='balanced', gamma=gamma, C=C)
    md_clf.fit(X, y) # apply SVM to training data and draw boundaries.
    
    return md_clf

def interpol_im(im, dim1 = 8, dim2 = 8, cmap = 'binary', grid_off = False):
    if((len(im.shape)) == 3):
        im = im[:, :, 0]
    
    x = np.arange(im.shape[1])
    y = np.arange(im.shape[0])
    
    f2d = interp2d(x, y, im)
    
    x_new = np.linspace(0, im.shape[1], dim1)
    y_new = np.linspace(0, im.shape[0], dim2)
    
    new_im = f2d(x_new, y_new)
    
    return new_im.flatten()
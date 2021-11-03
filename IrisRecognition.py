import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

from IrisLocalization import *
from IrisNormalization import *
from IrisEnhancement import *
from FeatureExtraction import *
from IrisMatching import *
from PerformanceEvaluation import *
import glob

def irisRecognition(imgs_train,imgs_test):
  size_train = len(imgs_train)
  size_test = len(imgs_test)
  # rotation degree
  degrees = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
  size_degrees = len(degrees) + 1 # because we also have the original one with degree 0 
  
  # create x training set
  x_train = np.zeros((size_train*size_degrees, 1536))
  k = 0 #index number
  for i in range(size_train):
    x_i, y_i, r_i, x_p, y_p, r_p = irisLocalization(imgs_train[i])
    normalized = irisNormalization(x_i, y_i, r_i, x_p, y_p, r_p, imgs_train[i])
    #enhanced = irisEnhancement(normalized)
    #v = featureExtraction(enhanced, 9)
    v = featureExtraction(normalized, 9)
    x_train[k] = v[1]
    for j in range(len(degrees)):
      k += 1
      normalized_rotate = imgRotate(normalized, degrees[j])
      #enhanced_rotate = irisEnhancement(normalized_rotate)
      #v_rotate = featureExtraction(enhanced_rotate, 9)
      v_rotate = featureExtraction(normalized_rotate, 9)
      x_train[k] = v_rotate[1]
    k += 1
  
  # create x test set
  x_test = np.zeros((size_test, 1536))
  for i in range(size_test):
    x_i, y_i, r_i, x_p, y_p, r_p = irisLocalization(imgs_test[i])
    normalized = irisNormalization(x_i, y_i, r_i, x_p, y_p, r_p, imgs_test[i])
    #enhanced = irisEnhancement(normalized)
    #v = featureExtraction(enhanced, 9)
    v = featureExtraction(normalized, 9)
    x_test[i] = v[1]
  
  # create y train set and test set
  y_train = np.repeat(np.linspace(1,108, num=108).astype(int), size_degrees*3)
  y_test = np.repeat(np.linspace(1,108, num=108).astype(int), 4)

  # create CRR graph, CRR table, ROC curve
  performanceEvaluation(x_train, y_train, x_test, y_test)

if __name__ == "__main__":
  imgs_train= [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob('datasets/CASIA Iris Image Database (version 1.0)/*/1/*.bmp'))]
  imgs_test= [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob('datasets/CASIA Iris Image Database (version 1.0)/*/2/*.bmp'))]
  irisRecognition(imgs_train,imgs_test)

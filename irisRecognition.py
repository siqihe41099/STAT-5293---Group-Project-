import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

from IrisLocalization  import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *
from IrisMatching      import *
from PerformanceEnvaluation import *

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
    #enhanced = ImageEnhancement(normalized)
    #v = featureExtraction(enhanced, 9)
    v = featureExtraction(normalized, 9)
    x_train[k] = v[1]
    for j in range(len(degrees)):
      k += 1
      normalized_rotate = imgRotate(normalized, degrees[j])
      #enhanced_rotate = ImageEnhancement(normalized_rotate)
      #v_rotate = featureExtraction(enhanced_rotate, 9)
      v_rotate = featureExtraction(normalized_rotate, 9)
      x_train[k] = v_rotate[1]
    k += 1
  
  # create x test set
  x_test = np.zeros((size_test, 1536))
  for i in range(size_test):
    x_i, y_i, r_i, x_p, y_p, r_p = irisLocalization(imgs_test[i])
    normalized = irisNormalization(x_i, y_i, r_i, x_p, y_p, r_p, imgs_test[i])
    #enhanced = ImageEnhancement(normalized)
    #v = featureExtraction(enhanced, 9)
    v = featureExtraction(normalized, 9)
    x_test[i] = v[1]
  
  # create y train set and test set
  y_train = np.repeat(np.linspace(1,108, num=108).astype(int), size_degrees*3)
  y_test = np.repeat(np.linspace(1,108, num=108).astype(int), 4)

  # create CRR graph, 
  performanceEvaluation(x_train, y_train, x_test, y_test)

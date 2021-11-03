from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import cv2
import numpy as np

def irisMatching(x_train, y_train, x_test, y_test, dimension=107,reduce=True):
  if reduce == False:
    metric=["manhattan","euclidean","cosine"]
    score = []
    for i in metric:
      classifier = NearestCentroid(metric=i)
      classifier.fit(x_train, y_train)
      score.append(classifier.score(x_test, y_test))
    return(score, x_train, x_test, classifier.centroids_)
  clf = lda(n_components=dimension)
  fit = clf.fit(x_train,y_train)
  X_train = fit.transform(x_train)
  X_test = fit.transform(x_test)
  metric=["l1","l2","cosine"]
  score = []
  for i in metric:
    classifier = NearestCentroid(metric=i)
    classifier.fit(X_train, y_train)
    score.append(classifier.score(X_test, y_test))
  return(score, X_train, X_test, classifier.centroids_)

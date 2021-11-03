from prettytable import PrettyTable
import cv2
import matplotlib.pyplot as plt
from IrisMatching import *
import scipy

from prettytable import PrettyTable
def performanceEvaluation(x_train, y_train, x_test, y_test):
  # Draw CRR curve
  dimension = [50,60,70,80,90,100,107]
  plt.figure()
  crr=[]
  for i in range(len(dimension)):
      crr.append(np.max(irisMatching(x_train, y_train, x_test, y_test, dimension[i])[0]))

  plt.plot(dimension, crr, color='darkorange',lw=1)
  plt.xlabel('Dimensionality of the feature vector')
  plt.ylabel('Correct recgnition rate')
  plt.scatter(dimension,crr,marker='*')
  plt.savefig("outputs/crr.png")
  plt.show()

  # Draw table
  scores_o, x_train, x_test, centroids = irisMatching(x_train, y_train, x_test, y_test, reduce=False)
  scores_r, x_train_r, x_test_r, centroids_r  = irisMatching(x_train, y_train, x_test, y_test, 100)
  table = PrettyTable()
  table.title = 'Recognition Results Using Different Similarity Measures'
  table.field_names = ['Similarity measure', 'Original feature set', 'Reduced feature set']
  table.add_row(['L1 distance measure', scores_o[0], scores_r[0]])
  table.add_row(['L2 distance measure', scores_o[1] , scores_r[1]])
  table.add_row(['Cosine distance measure', scores_o[2], scores_r[2]])
  print(table)

  # Draw ROC
  sim = np.zeros((x_test_r.shape[0], x_train_r.shape[0]))
  for i in range(sim.shape[0]): # 432
    for j in range(sim.shape[1]): # 108
      sim[i, j] = 1 - scipy.spatial.distance.cosine(x_test_r[i], x_train_r[j])
  fmr = []
  fnmr = []
  thresholds = np.linspace(0.4, 0.8, 20)
  for thres in thresholds:
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(sim.shape[0]):
      for j in range(sim.shape[1]):
        if sim[i,j] < thres:
          if y_test[i] == y_train[j]:
            fn += 1        
          else:
            tn += 1 
        else:  
          if y_test[i] != y_train[j]:
            fp += 1          
          else:
            tp += 1 
    fmr.append(fp/(fp+tn))
    fnmr.append(fn/(fn+tp))

  plt.plot(fmr, fnmr)
  plt.xlabel('False Match Rate')
  plt.ylabel('False Non-Match Rate')
  plt.savefig('outputs/roc.png')
  plt.show()

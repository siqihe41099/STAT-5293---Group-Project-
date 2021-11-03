import cv2
import numpy as np


def irisNormalization(x_i, y_i, r_i, x_p, y_p, r_p, img_in):
  # I_n is a M x N (64 x 512 in this experiment) normalized image
  M, N = 64, 512
  # Create an empty matrix to store the normalized image
  normalized = np.zeros((M, N))

  # thetas are the angles from 0 to 2*pi
  thetas = np.arange(N)/N * 2 * np.pi

  # Project the original iris into new coordinate system defined by LiMa's paper
  for i in range(M):
    for j in range(N):

      x_p_theta = x_p + r_p * np.cos(thetas[j]) 
      y_p_theta = y_p - r_p * np.sin(thetas[j]) 
      x_i_theta = x_p + r_i * np.cos(thetas[j]) 
      y_i_theta = y_p - r_i * np.sin(thetas[j]) 
      x, y = x_p_theta + (x_i_theta - x_p_theta) * i / M, y_p_theta + (y_i_theta - y_p_theta) * i / M
      
      # Check whether x,y exceeds the boundary, if it exceeds, then set the pixel value to the boundary pixel value.
      x = min(319, x) or max(0, x)
      y = min(279, y) or max(0, y)
      normalized[i, j] = img_in[int(y), int(x)]

  return(normalized)

#imgRotate is to calculate the rotated image
def imgRotate(img_in, degree):
  move = abs(int(512*degree/360))
  if degree > 0:
    return np.hstack([img_in[:,move:], img_in[:,:move]] )
  else:
    return np.hstack([img_in[:,(512 - move):], img_in[:,:(512 - move)]])

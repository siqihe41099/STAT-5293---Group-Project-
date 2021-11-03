import cv2
import numpy as np
import scipy.signal


# Defined fillter
def gabor(x, y, f, dx, dy):
  return(1/(2*np.pi*dx*dy)*np.exp(-1/2*((x**2/dx**2)+(y**2/dy**2)))*np.cos(2*np.pi*f*np.sqrt(x**2+y**2)))

import scipy.signal
def featureExtraction(img_histequal, kernal_size):
  # Define the region of interest 48X512
  # In this way, we can find regions closer to the pupil, which contain useful texture information for recognization
  img_roi = img_histequal[:48, :]
  # We have two channels based on sigma_x, simga_y
  # sigma_x, simga_y in first channel are 3 and 1.5, and the second channel 4.5 and 1.5. 
  dx1, dy1, dx2, dy2 = 3, 1.5, 4.5, 1.5
  # We also try different frequency: 1/deltaX, 1/deltaY
  # So we have four comibinations for defined filter kernel
  gabor_filter = np.zeros((4, kernal_size, kernal_size))
  kernal_ind = np.linspace(kernal_size-1, 0, kernal_size)-kernal_size//2
  for i in kernal_ind:
    for j in kernal_ind:
      gabor_filter[0, int(np.abs(j-kernal_size//2)), int(np.abs(i-kernal_size//2))] = gabor(i, j, 1/dx1, dx1, dy1)
      gabor_filter[2, int(np.abs(j-kernal_size//2)), int(np.abs(i-kernal_size//2))] = gabor(i, j, 1/dx2, dx2, dy2)
      gabor_filter[1, int(np.abs(j-kernal_size//2)), int(np.abs(i-kernal_size//2))] = gabor(i, j, 1/dy1, dx1, dy1)
      gabor_filter[3, int(np.abs(j-kernal_size//2)), int(np.abs(i-kernal_size//2))] = gabor(i, j, 1/dy2, dx2, dy2)
  # F denotes the filtered image
  F = [scipy.signal.convolve2d(img_roi,gabor_filter[0],mode='same'),
      scipy.signal.convolve2d(img_roi, gabor_filter[1],mode='same'),
      scipy.signal.convolve2d(img_roi, gabor_filter[2],mode='same'),
      scipy.signal.convolve2d(img_roi, gabor_filter[3],mode='same')]
  # Extract statistical features (mean and standard deviation) in each 8x8 small block of the filtered images
  v = [[], []]
  for k in range(4):
    i = 0
    while(i < img_roi.shape[0]):
      j = 0
      while(j < img_roi.shape[1]):
        m = 1/64*np.sum(np.abs(F[k][i:i+8, j:j+8]))
        v[k%2].append(m)
        s = 1/64*np.sum(np.abs(np.abs(F[k][i:i+8, j:j+8])-m))
        v[k%2].append(s)
        j = j+8
      i = i+8
  return(v)

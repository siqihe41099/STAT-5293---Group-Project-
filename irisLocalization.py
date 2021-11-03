import cv2
import numpy as np


def irisLocalization(img_in):
  # Since there are two graphs that has the min value at the edge, so we try to avoid this situation
  # We create a subset of original image
  imgsub = img_in[60:220,80:240]

  # project the image in the vertal and horizontal direction to 
  # approximately estimate the center of pupil
  # sum along vertical and horizontal directions
  h_projection = np.mean(imgsub, axis=0)
  v_projection = np.mean(imgsub, axis=1)
  # center coordinates of the pupil
  # the minimum of two projection profiles are considered as center coordinates of the pupil
  xp, yp = np.argmin(h_projection)+60, np.argmin(v_projection)+80
  if xp<60: 
    xp=60
  if yp<60:
    yp=60
  
  # Create a 120x120 region centered at (xp,yp) 
  img_sub = img_in[(yp-60):(yp+60), (xp-60):(xp+60)]
  #plt.imshow(img_sub)

  # Use HoughCircles to find the pupil
  # Blur the image to remove the effects of eyelash, eyelid and etc.
  img_blur = cv2.medianBlur(img_in,11)
  # Select threshold
  img_mask = cv2.inRange(img_blur, 0, 60)
  # Edge detection and binarize the image
  img_edge = cv2.Canny(img_mask, 100, 200)
  # Use HoughCircles to find the circles in the binary image
  pupils = cv2.HoughCircles(img_edge, cv2.HOUGH_GRADIENT, 10, 100)
  # Find the pupil
  if pupils is None: # If the previous threshold does not work
    img_sub = img_blur[(yp-60):(yp+60), (xp-60):(xp+60)]
    # Change a threshold
    _, thres = cv2.threshold(img_sub,0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Edge Detection:
    tight_p = cv2.Canny(thres, 30,20)
    #plt.imshow(tight_p)
    # Find the inner circle: pupil
    # Reference: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
    pupil = cv2.HoughCircles(tight_p, cv2.HOUGH_GRADIENT, 4, 300, minRadius=20, maxRadius=60)
    x_p, y_p, r_p = np.round(pupil[0,0]).astype('int')
    # the center of inner circle in the original plot
    x_p_o = xp-60+x_p
    y_p_o = yp-60+y_p
  if pupils is not None: # If the previous threshold works
    # To localize the pupil:
    # Step1: Find the center in the img_sub
    x_center = np.mean(img_sub, 0).argmin() + xp - 60 # x-value of new center in the original graph
    y_center = np.mean(img_sub, 1).argmin() + yp - 60 # y-value of new center in the original graph
    center_pupil = np.array([x_center, y_center], dtype=int) # the position of new center in the original graph
   
    # Find the circle whose center is closet to the center we found in the Step1
    dist_min = np.inf
    for pupil in pupils[0]:
      point = np.array([pupil[0], pupil[1]])
      dist = np.linalg.norm(center_pupil - point)
      if dist < dist_min:
        dist_min = dist
        circle_pupil = pupil.astype(int)
        x_p_o,y_p_o,r_p = circle_pupil
  output = img_in.copy()
  img = cv2.circle(output, (x_p_o, y_p_o), r_p, (255, 255, 255), 1)
  #plt.imshow(img, cmap="gray")


  # To localize the outerboundary:
  # bound the cornea region. We choose 130 according to He's paper
  top = np.max([0, (y_p_o-130)])
  bottom = np.min([280, (y_p_o+130)])
  left = np.max([0, (x_p_o-130)])
  right = np.min([320, (x_p_o+130)])
  img_sub2 = img_in[top:bottom, left:right]
  
  # Find the range of outer circle
  # Blurring first
  # Reference: https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
  blur = cv2.GaussianBlur(img_sub2, (7, 7), 0)
  # Use Edge detection (Canny Operator) to binarize the image
  # Reference: https://docs.opencv.org/3.4.15/d7/de1/tutorial_js_canny.html
  tight = cv2.Canny(blur, 30,20)
  subx = int(x_p_o - left)
  suby = int(y_p_o - top)
  subr = int(r_p)
  img_sub3 = tight.copy()

  # Cover the pupil size to avoid find the inner circle
  img_sub3[0:(suby+subr+20), (subx-subr-20):(subx+subr+20)] = 0
  # Find the outer circle by using HoughCircles
  iris = cv2.HoughCircles(img_sub3, cv2.HOUGH_GRADIENT,1,250,param1=30,param2=10,minRadius=98,maxRadius=118)
  if iris is not None:
    output = img_in.copy()
    # the center of outer circle in the original plot
    x_i, y_i, r_i = np.round(iris[0,0]).astype('int')
    x_i_o = x_i+left
    y_i_o = y_i+top
    img = cv2.circle(output, (x_i_o, y_i_o), r_i, (255, 255, 255), 1)
    img = cv2.circle(output, (x_p_o, y_p_o), r_p, (255, 255, 255), 1)
    # plt.imshow(img, cmap="gray")
    return(x_i_o, y_i_o, r_i, x_p_o, y_p_o, r_p)

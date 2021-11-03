def irisEnhancement(normalized):
  M, N = 64, 512
  # approximate intensity variation
  
  y=0
  var=[]
  #  The estimate of the background illumination consists of the mean of each 16X16 small block
  while(y<M):
    x=0
    while(x<N):
      temp_data = normalized[y:y+16, x:x+16]
      temp_var=np.mean(temp_data)
      var.append(temp_var)
      x=x+16
    y=y+16
  img_var=np.reshape(var,(np.int(M/16),np.int(N/16)))
  # Expand to the same size as normalized image by using bicubic interpolation
  # Reference: https://stackoverflow.com/questions/25975990/opencv2-bicubic-interpolation-while-resizing-image/25976162
  # Reference: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
  img_resize = cv2.resize(img_var, (N, M), interpolation = cv2.INTER_CUBIC)
  # Subtract the estimated background illumination from normalized image 
  # to remove the effects of illumination
  img_illu = normalized#-img_resize
  #img_illu = (img_illu-(np.min(img_illu)))/(np.max(img_illu)-np.min(img_illu))*255

  # Use histogram equalization to enhance the corrected image
  img_illu = img_illu.astype(np.uint8)
  # Create an empty matrix to store the enhanced image
  img_histequal = np.zeros((M, N))
  # Do histogram equalization on each 32X32 region
  y=0
  while(y<M):
    x=0
    while(x<N):
      img_histequal[y:y+32, x:x+32] = cv2.equalizeHist(img_illu[y:y+32, x:x+32])
      x=x+32
    y=y+32
  return(img_histequal)

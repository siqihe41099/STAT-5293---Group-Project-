# IrisRecognition
## Explain the whole logic of your design
In this model, we try to reproduce the model established by Li Ma in his paper "Personal Identification Based on Iris Texture Analysis". 
We divide the Iris Recognition into three parts consisting of six functions. <br />
1. Image Processing <br />
- Iris Localization
What we need to do first is to localize the iris in an eye image. According to Li's paper, to project the image in the verical and horizontal directions, then to find the minimum value of two axies (since pupil area is supposed to be the darkest area in the image). So we get xp,yp as the center of pupil. However, this center may not accurate for most of time. So we zoom in the image to focus on the pupil area. We create a 120x120 region centered at (xp,yp).<br />. 
To find the pupil: <br />. 
We use two different methods to find the right center and radius of pupil. We blur the image first to reduce the effects of eyelashes, eyelids and etc. The main method we use is to find the center in the 120x120 region first. After applying edge detection and HoughCircles to find circles in the blurred image, we compare the distance between the center of those circles found by HoughCircles and the center in the 120x120 region. The one which has the minimum distance to the center in the 120x120 region is the center of pupil. Since some pupils cannot be found because of the first threhold, we use another adaptive threshold. <br />. 
To find the outer boundary:  <br />. 
We bound the cornea region by creating a 130x130 region centered at the pupil's center. Then we gaussian blur the input image and utilize edge detection to binarize the cornea region. After it, we cover the pupil size to avoid HoughCircles() finds the inner circle. By setting the minRadius and maxRadius, HoughCircles() finds the outer circle's boundary. <br />. 
- Iris Normalization
Project the iris image into the new coordinate system given by Li to get the normalized image. After running the irisNormalization function, we will get 7 normzlied image by setting the initial angle values as -9, -6, -3, 0, 3, 6, and 9 degrees. In this way, we create more train sets which can make our model more robust.
- Image Enhancement
We need to consider the illumination effects to the normalized image. So we subtract estimated background illumination from the normalized image. Then use histogram equalization to improve the contrast of the image.
2. Feature Extraction
Define the region of interest as 48X512 size. In this way, we can find regions closer to the pupil, which contain useful texture information for recognization. Then we use defined filters to obtain the filtered image. Finally, we etract statistical features (mean and standard deviation) in each 8x8 small block of the filtered images to form vectors for later iris matching.
 3. Iris Matching
 We use LDA to reduce dimensionality and train the model. Then use the established model to classify the test dataset and check the correctness of our model.
## Briefly discuss the limitation(s) of the current design. How can you improve it?
1. There are two graphs that we cannot detect.
2. Our outer circle's boundary is not tight. It contains parts of the sclera area.
## Peer evaluation form


#### Iris Localization 
Weiyi Jiang: <br />
Dawei He: <br />
Siqi He: <br />
#### ImageEnhancement
Weiyi Jiang: <br />
Dawei He: <br />
Siqi He: <br />
#### FeatureExtraction
Weiyi Jiang: <br />
Dawei He: <br />
Siqi He: <br />
#### IrisMatching
Weiyi Jiang: <br />
Dawei He: <br />
Siqi He: <br />
#### PerformanceEvaluation
Weiyi Jiang: <br />
Dawei He: <br />
Siqi He: <br />
#### Others
Weiyi Jiang: <br />
Dawei He: <br />
Siqi He: <br />




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
We bound the cornea region by creating a 130x130 region centered at the pupil's center. Then we gaussian blur the input image and utilize edge detection to binarize the cornea region. After it, we cover the pupil size to avoid HoughCircles() finds the inner circle. By setting the minRadius and maxRadius, HoughCircles() finds the outer circle's boundary. <br />
- Iris Normalization <br />
Project the iris image into the new coordinate system given by Li to get the normalized image. After running the irisNormalization function, we will get 7 normzlied image by setting the initial angle values as different degrees. In this way, we create more train sets which can make our model more robust. Even though, based on the Li Ma's paper, the setting is -9, -6, -3, -1, 0, 3, 6, 9, according to our observation, we set the degrees as -5, -4, -3, -2, -1, 1, 2, 3, 4, 5.  <br />
- Image Enhancement <br />
We need to consider the illumination effects to the normalized image. So we subtract estimated background illumination from the normalized image. Then use histogram equalization to improve the contrast of the image. However, in our results, it shows that when we do not do hist equlization, the correcteness reaches the highest point.<br />
2. Feature Extraction <br />
Define the region of interest as 48X512 size. In this way, we can find regions closer to the pupil, which contain useful texture information for recognization. Then we use defined filters to obtain the filtered image. Finally, we etract statistical features (mean and standard deviation) in each 8x8 small block of the filtered images to form vectors for later iris matching.
 3. Iris Matching <br />
We use LDA to reduce dimensionality and train the model. Then use the established model to classify the test dataset and check the correctness of our model.
## Briefly discuss the limitation(s) of the current design. How can you improve it?
1. Our outer circle's boundary is not tight. It contains parts of the sclera area. It leads to the result that the normalization results is not so good. We haven't come up with an idea to solve it.
2. Hist Equalization doesn't work in our project, but if we do image quality accessment before image processing may solve the problem.
## Peer evaluation form
We did the group project and wrote those functions together. We all participated in brainstorming, searching materials, picking threshold and writing the functions. We did not separate the missions. Everyone in the group is responsible and willing to contribute to final work. 



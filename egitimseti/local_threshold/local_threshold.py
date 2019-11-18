#subdivided yapilmadi
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Fig1050(a)(sine_shaded_text_image).pgm",0)
#img = cv2.medianBlur(img,5)
kernel = np.ones((25,25),np.float32)/625
dst = cv2.filter2D(img,-1,kernel)
equ = cv2.equalizeHist(dst)

# global thresholding
ret1,th1 = cv2.threshold(img,170,255,cv2.THRESH_BINARY)

# Otsu's thresholding
#th2 =cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,811,0)
th2 =cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,0)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
th3 =cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
cv2.imwrite('reseultFig1050(a)(sine_shaded_text_image).pgm',images[8])

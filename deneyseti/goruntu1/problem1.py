"""
mehmet kucuk 121024075
gorunut isleme proje
1) esikleme
2)ayrit saptama 
3)morfolojic operator
4)baglanti bilesen analizi
5)cizgi/elips/daire tespiti 
))oznitelik cikartimi
"""
import numpy as np
import cv2
from skimage import measure
from imutils import contours
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
#subdivided yapilmadi
img = cv2.imread("problem1.jpg",0)
kernel = np.ones((5,5),np.float32)/25
kernelone = np.ones((35,35))##erotion icın secilen matris
dst = cv2.filter2D(img,-1,kernel)
equ = cv2.equalizeHist(dst)

# global thresholding
ret1,th1 = cv2.threshold(equ,170,255,cv2.THRESH_BINARY)

# Otsu's thresholding
th2 =cv2.adaptiveThreshold(equ,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,111,0)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# plot all the images and their histograms
images = [equ, 0, th1,
          equ, 0, th2,
          blur, 0, th3]
erosion = cv2.erode(images[8],kernelone,iterations = 1)#Gaussian filtering uygulanmis goruntuye otsu uyguladik tan sonra istenmeyen bolgeleri yok ettik
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
#bölgeleri adlandirip saydim
labels = measure.label(erosion, neighbors=8, background=0)
mask = np.zeros(erosion.shape, dtype="uint8")
 
# loop over the unique components
for label in np.unique(labels):
	# if this is the background label, ignore it
	if label == 0:
		continue
 
	# otherwise, construct the label mask and count the
	# number of pixels 
	labelMask = np.zeros(erosion.shape, dtype="uint8")
	labelMask[labels == label] = 255
	numPixels = cv2.countNonZero(labelMask)
 
	# if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
	if numPixels > 300:
		mask = cv2.add(mask, labelMask)
# find the contours in the mask, then sort them from left to
# right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]
 
# loop over the contours
for (i, c) in enumerate(cnts):
	# draw the bright spot on the image
	(x, y, w, h) = cv2.boundingRect(c)
	((cX, cY), radius) = cv2.minEnclosingCircle(c)
	cv2.circle(img, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
	cv2.putText(img, "#{}".format(i + 1), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
cv2.imwrite('resultp1.pgm',img)
# show the output image
#cv2.imshow("Image", img)
#cv2.waitKey(0)
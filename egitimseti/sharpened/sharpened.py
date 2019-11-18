import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal

import cv2

import numpy as np

img = cv2.imread("Fig0338(a)(blurry_moon).pgm",0)
kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
c=np.array([[0,0,0],[0,1,0],[0,0,0]])
dst = signal.convolve2d(img,kernel)
c=signal.convolve2d(img,c)

sharpened=c+dst
res = np.hstack((sharpened,dst))
cv2.imwrite('resultsharpened.pgm',res)

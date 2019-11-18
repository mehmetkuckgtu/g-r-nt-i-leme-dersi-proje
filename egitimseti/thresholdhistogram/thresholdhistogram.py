import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
from skimage.morphology import square
from skimage.filters.rank import threshold

import cv2

import numpy as np

img= cv2.imread("Fig1038(a)(noisy_fingerprint).pgm", 0)
c=np.array([[0,0,0],[0,1,0],[0,0,0]])
equ = cv2.equalizeHist(img)

ret1,th1 = cv2.threshold(equ,90,255,cv2.THRESH_BINARY)
images = [equ, 0, th1]

cv2.imwrite('resultFig1038(a)(noisy_fingerprint).pgm',images[2])
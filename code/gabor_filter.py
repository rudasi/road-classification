from skimage.util import img_as_float
from skimage import io
from skimage import color
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
import cv2

image = io.imread('..\data\\training\image_2\uu_000000.png')
image_ycbcr = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
image_gray = color.rgb2gray(image)
(a,b,c) = image_ycbcr.shape
image_y = np.zeros((a,b))
image_h = np.zeros((a,b))

image_hsv = color.rgb2hsv(image)

for i in range(a):
   for j in range(b):
      image_y[i][j] = image_ycbcr[i][j][0]
      image_h[i][j] = image_hsv[i][j][0] + image_ycbcr[i][j][0]

image_y = img_as_float(image_y)
image_y = (image_y - image_y.mean())/image_y.std()
image_h = img_as_float(image_h)
image_h = (image_h - image_h.mean())/image_h.std()
image_hsv_smooth = ndi.gaussian_filter(image_hsv, sigma=(1,1,1))

kernels = []
filtered = []
accum = np.zeros_like(image_y)

for theta in range(4):
   theta = theta/4.0 *np.pi
   for sigma in (0.5,1,2):
      for frequency in (0.1,0.4):
         kernel = gabor_kernel(frequency=frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
         fimg = ndi.convolve(image_y,kernel,mode='constant',cval=0)
         accum = np.minimum(accum,fimg,accum)
      #filtered.append(ndi.convolve(image_y,kernel,mode='wrap'))
      #kernels.append(kernel)

accum = (accum-accum.mean())/accum.std()

fig = plt.figure()
ax = fig.add_subplot(3,1,1)
ax.imshow(image)
ax = fig.add_subplot(3,1,2)
ax.imshow(image_ycbcr)
ax = fig.add_subplot(3,1,3)
ax.imshow(image_hsv)
plt.show()


import numpy as np
import cv2
from skimage import io
from skimage.util import img_as_float

def build_filters():
   filters = []
   ksize = 31
   for theta in np.arange(0, np.pi, np.pi/16):
      kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta,10.0,0.5,0,ktype=cv2.CV_32F)
      kern /= 1.5*kern.sum()
      filters.append(kern)
      return filters

def process(img, filters):
   accum = np.zeros_like(img)
   for kern in filters:
      fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
      np.maximum(accum, fimg, accum)
   return accum

if __name__ == '__main__':
   image = io.imread('..\data\\training\image_2\uu_000000.png')
   image_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
   image_ycbcr = img_as_float(image_ycbcr)
   a,b,c = image_ycbcr.shape
   img = np.zeros((a,b))
   for i in range(a):
      for j in range(b):
         img[i][j] = image_ycbcr[i][j][0]

   filters = build_filters()
   res1 = process(img, filters)
   cv2.imshow('result', res1)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

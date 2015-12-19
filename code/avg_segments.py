from skimage.segmentation import slic
from skimage.segmentation import boundaries
from skimage import io
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
print "starting"
image = io.imread('../data/training/image_2/uu_000000.png')
image_segment = slic(image, n_segments = 1000, sigma = 5)
t, indices = np.unique(image_segment, return_index=True)

image_flat = np.reshape(image,(1,image.shape[0]*image.shape[1],3))
image_segment_flat = np.reshape(image_segment,(1,image_segment.shape[0]*image_segment.shape[1]))

image_avg = []
print t.shape
print indices.shape
#print t
print "going into loop"
for j in indices[0:1]:
   print "indices: %d" %j
   val1 = 0.0
   val2 = 0.0
   val3 = 0.0
   for i in range(5):
      val1 += image_flat[0][j+i][0]*1.0
      val2 += image_flat[0][j+i][1]*1.0
      val3 += image_flat[0][j+i][2]*1.0


   print val1
   val1 /= 5
   print "avg: %f" %(val1)
   val2 /= 5
   val3 /= 5
   image_avg.append([val1,val2,val3])

#image_final = np.reshape(image_avg,(image.shape[0],image.shape[1],3))
print image_avg

#packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

uu_num = 98
numSegments = 200
images = []
image_indices = []
image_segments = []

for i in range(uu_num):
   print "data %d" %(i+1)
   img = ''
   if i < 10:
      img = '0' + str(i)
   else:
      img = str(i)
   image = img_as_float(io.imread('..\data\\training\image_2\uu_0000' + img + '.png'))
   images.append(image)
   segments = slic(image, n_segments = numSegments, sigma = 5)
   image_segments.append(segments)
   a, indices = np.unique(segments, return_index=True)
   image_indices.append(indices)

np.savez('uu_data_images.npz',*[images[i] for i in range(uu_num)])
np.savez('uu_data_segments.npz',*[image_segments[i] for i in range(uu_num)])
np.savez('uu_data_indices.npz',*[image_indices[i] for i in range(uu_num)])


'''
data_images = np.load('uu_data_images.npz')
data_segments = np.load('uu_data_segments.npz')
data_indices = np.load('uu_data_indices.npz')

print "image 1"
print data_indices['arr_0']
print "image 2"
print data_indices['arr_1']
print "image 3"
print data_indices['arr_2']

fig = plt.figure()
ax = fig.add_subplot(3,1,1)
ax.imshow(mark_boundaries(data_images['arr_0'],data_segments['arr_0']))

ax = fig.add_subplot(3,1,2)
ax.imshow(mark_boundaries(data_images['arr_1'],data_segments['arr_1']))

ax = fig.add_subplot(3,1,3)
ax.imshow(mark_boundaries(data_images['arr_2'],data_segments['arr_2']))

plt.axis("off")
plt.show()
'''

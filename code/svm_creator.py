#packages
from sklearn import svm
from skimage.util import img_as_float
from skimage import io
import numpy as np
import pickle

uu_num = 70
numSegments = 200

images = []
image_indices = []
gt_images = []

for i in range(uu_num):
   print "data %d" %(i+1)
   img = ''
   if i < 10:
      img = '0' + str(i)
   else:
      img = str(i)

   gt_image = img_as_float(io.imread('..\data\\training\gt_image_2\uu_road_0000' + img + '.png'))
   gt_images.append(image)


print "reading indices"
data_indices = np.load('uu_data_indices.npz')
print "reading training images"
data_images = np.load('uu_data_images.npz')

X = []
y = []

for i in range(uu_num):
   X.append([data_images[i] for i in data_indices])
   y.append([gt_images[i] for i in data_indices])

X = np.asarray(X)
y = np.asarray(y)

print X.flags
print y.flags

svm_model = svm.SVC()
svm.fit(X,y)
svm.support_vectors_

output = open('svm_model.pkl','wb')
pickle.dump(svm, output)
output.close()

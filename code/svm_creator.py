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
data_images = []

for i in range(uu_num):
   print "data %d" %(i+1)
   img = ''
   if i < 10:
      img = '0' + str(i)
   else:
      img = str(i)
   image = img_as_float(io.imread('..\data\\training\image_2\uu_0000' + img + '.png'))
   image = image.flatten()
   data_images.append(image)
   gt_image = img_as_float(io.imread('..\data\\training\gt_image_2\uu_road_0000' + img + '.png'))
   gt_image = gt_image.flatten()
   gt_images.append(gt_image)


print "reading indices"
data_indices = np.load('uu_data_indices.npz')
#print "reading training images"
#data_images = np.load('uu_data_images.npz')

X = []
y = []

for i in range(uu_num):
   X.append([data_images[i][j] for j in data_indices['arr_'+str(i)]])
   y.append([gt_images[i][j] for j in data_indices['arr_'+str(i)]])

X = np.asarray(X)
X = X.flatten()
X = [[i] for i in X]
X = np.asarray(X)
y = np.asarray(y)
y = y.flatten()
y = [[i] for i in y]
y = np.asarray(y)

print "X flags"
print X.flags
print "X shape"
print X.shape
print "y flags"
print y.flags
print "y shape"
print y.shape

print "training svm model"
svm_model = svm.SVC()
svm_model.fit(X,y)
svm_model.support_vectors_
print "done training svm model"

output = open('svm_model.pkl','wb')
pickle.dump(svm_model, output)
output.close()

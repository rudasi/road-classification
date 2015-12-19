#packages
from sklearn import metrics
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn import svm
from skimage.util import img_as_float
from skimage import io
from skimage import color
import numpy as np
import pickle

uu_num_train = 7 
uu_num_test = 2 
#uu_num_test = 1
numSegments = 2000

#training images with superpixels rgb
images_train = []
#training images with superpixels hsv
images_train_hsv = []
#gt training images with pixel indexed by image_train_indices
gt_images_train = []
#training images indices of first value in superpixel
images_train_indices = []
test_images_predict = []
test_gt_images = []
data_images = []
#test images, whole (no superpixels for now)
images_test = []

#classifier model   
#svm_classifier = svm.SVC()

#This function reads data for training and creates superpixels.
#The superpixels are stored in images_train and gt_images_train
def slic_data():
   for i in range(uu_num_train+uu_num_test):

      print "data %d" %(i+1)
      img_name = ''
      if i < 10:
         img_name = '0' + str(i)
      else:
         img_name = str(i)

      #Read first 70 images as floats
      img = img_as_float(io.imread('..\data\\training\image_2\uu_0000' + img_name + '.png'))
      img_hsv = color.rgb2hsv(img)
      gt_img = img_as_float(io.imread('..\data\\training\gt_image_2\uu_road_0000' + img_name + '.png'))
      
      #Create superpixels for training images
      image_segment = slic(img, n_segments = numSegments, sigma = 5)
      t, train_indices = np.unique(image_segment, return_index=True) 
      images_train_indices.append(train_indices)
      image = np.reshape(img,(1,(img.shape[0]*img.shape[1]),3))
      image_hsv = np.reshape(img_hsv,(1,(img_hsv.shape[0]*img_hsv.shape[1]),3))
      #images_train.append([image[0][i] for i in train_indices]) 
      images_train_hsv.append([image_hsv[0][i] for i in train_indices]) 
 
      #Create gt training image values index at train_indices and converted to 1 or 0
      gt_image = np.reshape(gt_img, (1,(gt_img.shape[0]*gt_img.shape[1]),3))
      gt_image = [1 if gt_image[0][i][2] > 0 else 0 for i in train_indices]
      gt_images_train.append(gt_image)
 

def svm_run(svm_classifier):
   print "svm run"
   X = []
   y = []

   for i in range(uu_num_train):
      for j in range(len(images_train_hsv[i])):
         #val = np.zeros(6)
         #val[0:3] = images_train[i][j]
         #val[3:6] = images_train_hsv[i][j]
         #X.append(val)

        # X.append(images_train[i][j])
         X.append(images_train_hsv[i][j])
         y.append(gt_images_train[i][j])

   X = np.asarray(X)
   y = np.asarray(y)
   print "X.shape = %s y.shape = %s" %(X.shape,y.shape)
   svm_classifier.fit(X,y)

   output = open('svm_model.pkl','wb')
   pickle.dump(svm_classifier, output)
   output.close()
   
   print "The support vectors lenght are:"
   print len(svm_classifier.support_vectors_)

def svm_predict(case,svm_classifier):
   print "svm predict"
   A = []
   b = []
 
   if case == 1:
      for i in range(uu_num_test):
         img_name = i + uu_num_train
         if img_name < 10:
            img_name = '0' + str(img_name)
         else:   
            img_name = str(img_name)
         print "data %d " %(i+uu_num_train)
         #Read test images as floats
         img = img_as_float(io.imread('..\data\\training\image_2\uu_0000' + img_name + '.png'))
         gt_img = img_as_float(io.imread('..\data\\training\gt_image_2\uu_road_0000' + img_name + '.png'))
      
         #Create superpixels for test images
         image_segment = slic(img, n_segments = numSegments, sigma = 5)
         t, train_indices = np.unique(image_segment, return_index=True) 
         images_train_indices.append(train_indices)
         image = np.reshape(img,(1,(img.shape[0]*img.shape[1]),3))
         images_train.append([image[0][i] for i in train_indices]) 
      
         #Create gt test image values index at train_indices and converted to 1 or 0
         gt_image = np.reshape(gt_img, (1,(gt_img.shape[0]*gt_img.shape[1]),3))
         gt_image = [1 if gt_image[0][i][2] > 0 else 0 for i in train_indices]
         print "len of gt_image: %d with ones: %d" %(len(gt_image),gt_image.count(1)) 
         
         gt_images_train.append(gt_image)
      
      for i in range(uu_num_test):
         for j in range(len(images_train[i])):
            A.append(images_train[i][j])
            b.append(gt_images_train[i][j])
   
   else:
      for i in range(uu_num_train,uu_num_train+uu_num_test):
         for j in range(len(images_train_hsv[i])):
            #val = np.zeros(6)
            #val[0:3] = images_train[i][j]
            #val[3:6] = images_train_hsv[i][j]
            #A.append(val)
            #A.append(images_train[i][j])
            A.append(images_train_hsv[i][j])
            b.append(gt_images_train[i][j])

   A = np.asarray(A)
   b = np.asarray(b)
   print "A.shape = %s, b.shape = %s" %(A.shape,b.shape)

   predicted = svm_classifier.predict(A)
   print svm_classifier.score(A,b)

   #for i in range(len(gt_images_train)):
   #   for j in range(len(gt_images_train[i])):
   #      print "%d, %d" %(gt_images_train[i][j], predicted[j]) 

   print("Classification report for classifier %s:\n%s\n" %(svm_classifier,metrics.classification_report(b,predicted)))


if __name__=="__main__": 
   svm_classifier = svm.SVC(kernel='rbf', class_weight={0:1,1:1.5})
   #svm_classifier = svm.NuSVC(nu=0.4,kernel='rbf')
   slic_data()   
   svm_run(svm_classifier)
   svm_predict(0,svm_classifier)
   '''
   with open('svm_model.pkl','rb') as handle:
      svm_classifier = pickle.load(handle)
     # svm_classfier = s
      print "svm classifier"
      print svm_classifier
      svm_predict(1,svm_classifier)
   '''
#output = open('svm_model.pkl','wb')
#pickle.dump(svm_model, output)
#output.close()

#packages
from sklearn import metrics
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn import svm
from skimage.util import img_as_float
from skimage import io
from skimage import color
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
import numpy as np
import pickle
import cv2
import random

numSegments = 1000

#training images with superpixels rgb
images_train = []
#training images with superpixels filtered
images_train_filtered = []
#training images with superpixels hsv
images_train_hsv = []
#training images with superpixels ycbcr
images_train_ycbcr = []
#gt training images with pixel indexed by image_train_indices
gt_images_train = []
#training images indices of first value in superpixel
images_train_indices = []
#image files
input_files = []
output_files = []
total_num_files = 289
training = 172
validation = 30
testing = 87
training_files = []
training_files_gt = []
validation_files = []
validation_files_gt = []
testing_files = []
testing_files_gt = []
uu_num_train = training
uu_num_valid = validation
uu_num_test = testing

def create_files():
   global input_files
   global output_files
   global training_files
   global training_files_gt
   global validation_files
   global validation_files_gt
   global testing_files
   global test_files_gt

   for i in range(95):
      img = ''
      if i < 10:
         img = '0'+str(i)
      else:
         img = str(i)
      input_files.append('../data/training/image_2/um_0000'+img+'.png')
      output_files.append('../data/training/gt_image_2/um_road_0000'+img+'.png')
   for i in range(96):
      img = ''
      if i < 10:
         img = '0'+str(i)
      else:
         img = str(i)
      input_files.append('../data/training/image_2/umm_0000'+img+'.png')
      output_files.append('../data/training/gt_image_2/umm_road_0000'+img+'.png')

   for i in range(98):
      img = ''
      if i < 10:
         img = '0'+str(i)
      else:
         img = str(i)
      input_files.append('../data/training/image_2/uu_0000'+img+'.png')
      output_files.append('../data/training/gt_image_2/uu_road_0000'+img+'.png')
   temp = random.sample(range(total_num_files),training+validation+testing)
   for i in temp[:training]:
      training_files.append(input_files[i])
      training_files_gt.append(output_files[i])
   for i in temp[training:training+validation]:
      validation_files.append(input_files[i])
      validation_files_gt.append(output_files[i])
   for i in temp[training+validation:training+validation+testing]:
      testing_files.append(input_files[i])
      testing_files_gt.append(output_files[i])

def initialise_lists():
    global images_train
    global images_train_filtered
    global images_train_hsv
    global images_train_ycbcr
    global gt_images_train
    global images_train_indices

    images_train = []
    images_train_filtered = []
    images_train_hsv = []
    images_train_ycbcr = []
    gt_images_train = []
    images_train_indices = []

#The superpixels are stored in images_train and gt_images_train
def slic_data():
   global images_train_hsv
   global images_train_filtered
   global images_train_indices
   global gt_images_train

   for i in range(uu_num_train+uu_num_valid+uu_num_test):
      #print "data %d" %(i+1)
      img_name = ''
      if i < 10:
         img_name = '0' + str(i)
      else:
         img_name = str(i)

      #Read first 70 images as floats
      #img = io.imread('../data/training/image_2/uu_0000' + img_name + '.png')
      if i < uu_num_train:
         img = io.imread(training_files[i])
         gt_img = img_as_float(io.imread(training_files_gt[i]))
         print "training"
         print i+1, training_files[i], training_files_gt[i]
      elif i >= uu_num_train and i < uu_num_train+uu_num_valid:
         img = io.imread(validation_files[i-uu_num_train])
         gt_img = img_as_float(io.imread(validation_files_gt[i-uu_num_train]))
         print "validation"
         print i+1, validation_files[i-uu_num_train], validation_files_gt[i-uu_num_train]
      elif i >= uu_num_train+uu_num_valid and i < uu_num_train+uu_num_valid+uu_num_test:
         img = io.imread(testing_files[i-uu_num_train-uu_num_valid])
         gt_img = img_as_float(io.imread(testing_files_gt[i-uu_num_train-uu_num_valid]))
         print "testing"
         print i+1, testing_files[i-uu_num_train-uu_num_valid], testing_files_gt[i-uu_num_train-uu_num_valid]
      #gt_img = img_as_float(io.imread('../data/training/gt_image_2/uu_road_0000' + img_name + '.png'))
      img_hsv = color.rgb2hsv(img)
      img_ycbcr = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
      img_ycbcr = img_as_float(img_ycbcr)
      (a,b,c) = img_ycbcr.shape
      img_y = np.zeros((a,b))

      for n in range(a):
         for m in range(b):
            img_y[n][m] = img_ycbcr[n][m][0]
      img_filtered = np.zeros_like(img_y)
      for theta in range(4):
         theta = theta/4.0 * np.pi
         for sigma in (0.1,0.3):
            for frequency in (0.2,0.3,0.5,0.6):
               kernel = np.real(gabor_kernel(frequency = frequency, theta = theta, sigma_x =sigma, sigma_y=sigma))
               val = ndi.convolve(img_y, kernel, mode='wrap')
               img_filtered = np.maximum(img_filtered,val,img_filtered)
      #img_filtered = (img_filtered-img_filtered.mean())/img_filtered.std()

      img_filtered /= img_filtered.max()
      img_filtered = img_filtered.flatten()

      #Create superpixels for training images
      image_segment = slic(img, n_segments = numSegments, sigma = 5)
      t, train_indices = np.unique(image_segment, return_index=True)
      images_train_indices.append(train_indices)
      #image = np.reshape(img,(1,(img.shape[0]*img.shape[1]),3))
      #img_hsv = ndi.gaussian_filter(img_hsv, sigma=(1,1,1))
      image_hsv = np.reshape(img_hsv,(1,(img_hsv.shape[0]*img_hsv.shape[1]),3))
      val1 = 0.0
      val2 = 0.0
      val3 = 0.0
      gabor_val = 0.0
      temp = []
      temp_gabor = []
      for k in train_indices[:950]:
         val1 = 0.0
         val2 = 0.0
         val3 = 0.0
         gabor_val = 0.0
         for l in range(10):
            val1 += image_hsv[0][k+l][0]*1.0
            val2 += image_hsv[0][k+l][1]*1.0
            val3 += image_hsv[0][k+l][2]*1.0
            gabor_val += img_filtered[k+l]
         val1 /= 10
         val2 /= 10
         val3 /= 10
         gabor_val /= 10

         temp.append([val1,val2,val3])
         temp_gabor.append(gabor_val)
      images_train_hsv.append(temp)
      #images_train_hsv[j] = temp
      #image_ycbcr = np.reshape(img_ycbcr,(1,(img_ycbcr.shape[0]*img_ycbcr.shape[1]),3))
      images_train_filtered.append(temp_gabor)
      #images_train_filtered.append([img_filtered[i] for i in train_indices])
      #images_train.append([image[0][i] for i in train_indices])
      #images_train_ycbcr.append([image_ycbcr[0][i] for i in train_indices])
      #images_train_hsv.append([image_hsv[0][i] for i in train_indices])

      #Create gt training image values index at train_indices and converted to 1 or 0
      gt_image = np.reshape(gt_img, (1,(gt_img.shape[0]*gt_img.shape[1]),3))
      gt_image = [1 if gt_image[0][z][2] > 0 else 0 for z in train_indices]
      gt_images_train.append(gt_image)
   images_train_hsv = np.asarray(images_train_hsv)
   images_train_filtered = np.asarray(images_train_filtered)

def svm_run(svm_classifier):
   global images_train_hsv
   global images_train_filtered
   global gt_images_train
   print "svm run"
   X = []
   y = []
   print images_train_hsv.shape
   print images_train_filtered.shape
   for i in range(uu_num_train):
      for j in range(images_train_hsv.shape[1]):
         val = np.zeros(4)
         val[0:3] = images_train_hsv[i][j]
         val[3] = images_train_filtered[i][j]
         X.append(val)

        # X.append(images_train[i][j])
         #X.append(images_train_hsv[i][j])
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

def svm_predict(svm_classifier):
   global images_train_hsv
   global gt_images_train
   global images_train_filtered
   print "svm predict"
   A = []
   b = []

   for i in range(uu_num_train,uu_num_train+uu_num_valid):
      for j in range(len(images_train_hsv[i])):
         val = np.zeros(4)
         val[0:3] = images_train_hsv[i][j]
         val[3] = images_train_filtered[i][j]
         A.append(val)
         #A.append(images_train_hsv[i][j])
         b.append(gt_images_train[i][j])

   A = np.asarray(A)
   b = np.asarray(b)
   print "A.shape = %s, b.shape = %s" %(A.shape,b.shape)
   print "validation results"
   predicted = svm_classifier.predict(A)

   #for i in range(len(gt_images_train)):
   #   for j in range(len(gt_images_train[i])):
   #      print "%d, %d" %(gt_images_train[i][j], predicted[j])

   print("Classification report for classifier on validation %s:\n%s\n" %(svm_classifier,metrics.classification_report(b,predicted)))

   A = []
   b = []

   for i in range(uu_num_train+uu_num_valid,uu_num_train+uu_num_valid+uu_num_test):
      for j in range(len(images_train_hsv[i])):
         val = np.zeros(4)
         val[0:3] = images_train_hsv[i][j]
         val[3] = images_train_filtered[i][j]
         A.append(val)
         #A.append(images_train_hsv[i][j])
         b.append(gt_images_train[i][j])

   A = np.asarray(A)
   b = np.asarray(b)
   print "A.shape = %s, b.shape = %s" %(A.shape,b.shape)
   print "testing results"
   predicted = svm_classifier.predict(A)
   print("Classification report for classifier on testing %s:\n%s\n" %(svm_classifier,metrics.classification_report(b,predicted)))


if __name__=="__main__":
   create_files()
   #weights = [0.8,1,1.2,1.4,1.6,1.8,2.0,2.4,2.6,3]
   #weights from precision recall graph
   weights = [1.14]
   for w in weights:
      print "Running with weight w:%f" %w
      initialise_lists()
      svm_classifier = svm.SVC(kernel='rbf',cache_size=500,
      class_weight={0:1,1:w})
      slic_data()
      svm_run(svm_classifier)
      svm_predict(svm_classifier)
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

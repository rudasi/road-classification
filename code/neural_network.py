#packages
from sklearn import metrics
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from sklearn.neural_network import MLPClassifier
from skimage.util import img_as_float
from skimage import io
from skimage import color
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
import numpy as np
import cv2
import random

numSegments = 1000

#training images with superpixels filtered
images_train_filtered = []
images_train_filtered_flat = []
#training images with superpixels hsv
images_train_hsv = []
images_train_hsv_flat = []
#gt training images with pixel indexed by image_train_indices
gt_images_train = []
gt_images_flat = []
#training images indices of first value in superpixel
images_train_indices = []

#image files
input_files = []
output_files = []
total_num_files = 289
'''
training = 172
validation = 30
testing = 87
'''
training = 172
validation = 30
testing = 87

training_files = []
training_files_gt = []
validation_files = []
validation_files_gt = []
testing_files = []
testing_files_gt = []
indices1s = []
indices0s = []
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
   global images_train_filtered
   global images_train_hsv
   global gt_images_train
   global images_train_indices

   images_train = []
   images_train_filtered = []
   images_train_filtered_flat = []
   images_train_hsv = []
   images_train_hsv_flat = []
   images_train_ycbcr = []
   gt_images_train = []
   gt_images_flat = []
   images_train_indices = []

#Superpixel maker
def slic_data():
   global images_train_filtered
   global images_train_hsv
   global gt_images_train
   global images_train_indices
   global indices1s
   global indices0s
   global images_train_filtered_flat
   global images_train_hsv_flat
   global gt_images_flat

   print "starting slic data"
   for i in range(uu_num_train+uu_num_valid+uu_num_test):
      #print "data %d" %(i+1)
      img_name = ''
      if i < 10:
         img_name = '0' + str(i)
      else:
         img_name = str(i)

      if i < uu_num_train:
         img = io.imread(training_files[i])
         gt_img = img_as_float(io.imread(training_files_gt[i]))
         #print "training"
         #print i+1, training_files[i], training_files_gt[i]
      elif i >= uu_num_train and i < uu_num_train+uu_num_valid:
         img = io.imread(validation_files[i-uu_num_train])
         gt_img = img_as_float(io.imread(validation_files_gt[i-uu_num_train]))
         #print "validation"
         #print i+1, validation_files[i-uu_num_train], validation_files_gt[i-uu_num_train]
      elif i >= uu_num_train+uu_num_valid and i < uu_num_train+uu_num_valid+uu_num_test:
         img = io.imread(testing_files[i-uu_num_train-uu_num_valid])
         gt_img = img_as_float(io.imread(testing_files_gt[i-uu_num_train-uu_num_valid]))
         #print "testing"
         #print i+1, testing_files[i-uu_num_train-uu_num_valid], testing_files_gt[i-uu_num_train-uu_num_valid]

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
               kernel = np.real(gabor_kernel(frequency = frequency, theta = theta, sigma_x=sigma,sigma_y=sigma))
               val = ndi.convolve(img_y, kernel, mode='wrap')
               img_filtered = np.maximum(img_filtered, val, img_filtered)
      img_filtered /= img_filtered.max()
      img_filtered = img_filtered.flatten()

      #Create superpixels for training images
      image_segment = slic(img, n_segments = numSegments, sigma = 5)
      t, train_indices = np.unique(image_segment, return_index=True)
      images_train_indices.append(train_indices)
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
         for l in range(20):
            val1 += image_hsv[0][k+l][0]*1.0
            val2 += image_hsv[0][k+l][1]*1.0
            val3 += image_hsv[0][k+l][2]*1.0
            gabor_val += img_filtered[k+l]
         val1 /= 20
         val2 /= 20
         val3 /= 20
         gabor_val /= 20

         temp.append([val1,val2,val3])
         temp_gabor.append(gabor_val)
      images_train_hsv.append(temp)
      images_train_filtered.append(temp_gabor)
      gt_image = np.reshape(gt_img, (1,(gt_img.shape[0]*gt_img.shape[1]),3))
      gt_image = [1 if gt_image[0][z][2] > 0 else 0 for z in train_indices[:950]]
      gt_images_train.append(gt_image)

   gt_images_train = np.asarray(gt_images_train)
   print gt_images_train.shape
   gt_images_flat = np.reshape(gt_images_train,(1,(gt_images_train.shape[0]*   gt_images_train.shape[1])))
   print gt_images_flat.shape
   indices1s = [index for index,value in enumerate(gt_images_flat[0]) if value == 1]
   indices0s = [index for index,value in enumerate(gt_images_flat[0]) if value == 0]
   print len(indices1s)
   print len(indices0s)
   images_train_hsv = np.asarray(images_train_hsv)
   images_train_hsv_flat = np.reshape(images_train_hsv,(1,(images_train_hsv.shape[0]*images_train_hsv.shape[1]),3))
   images_train_filtered = np.asarray(images_train_filtered)
   images_train_filtered_flat = np.reshape(images_train_filtered,(1,images_train_filtered.shape[0]*images_train_filtered.shape[1]))

def nn_run(nn_classifier,A,b):
   '''
   global images_train_hsv
   global images_train_filtered
   global gt_images_train
   print "nn run"
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
         y.append(gt_images_train[i][j])
   X = np.asarray(X)
   y = np.asarray(y)
   '''
   #print "X.shape = %s y.shape = %s" %(X.shape,y.shape)
   nn_classifier.fit(A,b)
   print "number of outputs %d" %nn_classifier.n_outputs_

def nn_predict(nn_classifier,A,b):
   '''
   global images_train_hsv
   global gt_images_train
   global images_train_filtered

   A = []
   b = []

   for i in range(uu_num_train, uu_num_train+uu_num_valid):
      for j in range(len(images_train_hsv[i])):
         val = np.zeros(4)
         val[0:3] = images_train_hsv[i][j]
         val[3] = images_train_filtered[i][j]
         A.append(val)
         b.append(gt_images_train[i][j])

   A = np.asarray(A)
   b = np.asarray(b)
   print "A.shape = %s, b.shape = %s" %(A.shape, b.shape)
   print "validation results"
   '''
   predicted = nn_classifier.predict(A)

   print("Classification report for nn classifier on validation %s:\n%s\n" %(nn_classifier, metrics.classification_report(b,predicted)))
   return predicted
'''
   A = []
   b = []

   for i in range(uu_num_train+uu_num_valid, uu_num_train+uu_num_valid+uu_num_test):
      for j in range(len(images_train_hsv[i])):
         val = np.zeros(4)
         val[0:3] = images_train_hsv[i][j]
         val[3] = images_train_filtered[i][j]
         A.append(val)
         b.append(gt_images_train[i][j])

   A = np.asarray(A)
   b = np.asarray(b)
   print "A.shape = %s, b.shape = %s" %(A.shape, b.shape)
   print "testing results"
   predicted = nn_classifier.predict(A)

   print("Classification report for nn classifier on testing %s:\n%s\n" %(nn_classifier, metrics.classification_report(b,predicted)))
'''

if __name__=="__main__":
   create_files()
   initialise_lists()
   slic_data()

   b = []
   A = []
   gty = []
   gtx = []
   num_classifiers = len(indices0s)/len(indices1s)

   for i in indices1s:
      gty.append(gt_images_flat[0][i])
      gtx_temp = images_train_hsv_flat[0][i].tolist()
      gtx_temp.append(images_train_filtered_flat[0][i])
      gtx.append(gtx_temp)

   for i in range(num_classifiers-1):
      tempy = []
      tempx = []
      for j in range(len(indices1s)):
         tempy.append(gt_images_flat[0][indices0s[i*len(indices1s)+j]])
         tempx_temp = images_train_hsv_flat[0][indices0s[i*len(indices1s)+j]].tolist()
         tempx_temp.append(images_train_filtered_flat[0][indices0s[i*len(indices1s)+j]])
         tempx.append(tempx_temp)

      b.append(gty+tempy)
      A.append(gtx+tempx)

   leftover = len(indices0s)-(num_classifiers*len(indices1s))
   tempy = []
   tempx = []

   for i in range(leftover):
      tempy.append(gt_images_flat[0][indices0s[(num_classifiers-1)*len(indices1s)+i]])
      tempx_temp = images_train_hsv_flat[0][indices0s[(num_classifiers-1)*len(indices1s)+i]].tolist()
      tempx_temp.append(images_train_filtered_flat[0][indices0s[(num_classifiers-1)*len(indices1s)+i]])
      tempx.append(tempx_temp)
   b.append(gty+tempy)
   A.append(gtx+tempx)

   for hp in [150]:
      print "hyperparameters layers %s" %hp
      classifiers = []
      for i in range(num_classifiers):
         classifiers.append(MLPClassifier(activation='relu', algorithm='l-bfgs', alpha=0.1, hidden_layer_sizes=(5,), max_iter = hp, validation_fraction=0.15, random_state=1))

      sXpredictor = []
      sypredictor = []
      sXtesting = []
      sytesting = []

      for i in range(num_classifiers):
         print "classifier %d training" %(i+1)
         t = int(((training*1.0)/(training+validation+testing))*len(A[i]))
         v = int(((training+validation*1.0)/(training+validation+testing))*len(A[i]))
         X = A[i]
         y = b[i]
         c = zip(X,y)
         random.shuffle(c)
         sX = []
         sy = []
         for k in c:
            sX.append(k[0])
            sy.append(k[1])
         sXpredictor.append(sX[t:v])
         sypredictor.append(sy[t:v])
         sXtesting += sX[v:]
         sytesting += sy[v:]

         count1train = len([index for index,value in enumerate(sy[:t]) if value == 1])
         count0train = len([index for index,value in enumerate(sy[:t]) if value == 0])
         count1valid = len([index for index,value in enumerate(sy[t:]) if value == 1])
         count0valid = len([index for index,value in enumerate(sy[t:]) if value == 0])
         print "vals are %d, %d, %d, %d" %(count1train,count0train,count1valid,count0valid)
         nn_run(classifiers[i],sX[:t],sy[:t])
      for i in range(num_classifiers):
         print "classfier %d validation prediction" %(i+1)
         nn_predict(classifiers[i],sXpredictor[i],sypredictor[i])
      print "size of testing data is %d" % len(sytesting)

      results = []
      for i in range(num_classifiers):
         results.append(nn_predict(classifiers[i],sXtesting,sytesting))

      voted_result = [0] * len(sytesting)

      for i in range(len(sytesting)):
         for j in range(num_classifiers-1):
            voted_result[i] += results[j][i]
         voted_result[i] = voted_result[i]/((num_classifiers-1)*1.0)

      voted_result = [round(val) for val in voted_result]
      print("Classification report for nn classifier on testing average %s:\n%s\n" %(classifiers[0], metrics.classification_report(sytesting,voted_result)))


import numpy as np
import matplotlib.pyplot as plt
import math

c1weights = [0.5,0.6,0.8,1.0,1.5,2.0]

class0precision_w = {0.5:[0.81,0.82,0.81,0.83],
                     0.6:[0.84,0.82,0.82,0.82],
                     0.8:[0.85,0.83,0.84,0.83],
                     1.0:[0.86,0.85,0.86,0.85],
                     1.5:[0.92,0.91,0.91,0.88],
                     2.0:[0.95,0.95,0.95,0.93]}

class0recall_w =    {0.5:[0.92,0.93,0.93,0.93],
                     0.6:[0.91,0.91,0.92,0.92],
                     0.8:[0.89,0.89,0.90,0.91],
                     1.0:[0.88,0.87,0.88,0.89],
                     1.5:[0.83,0.81,0.81,0.84],
                     2.0:[0.77,0.77,0.76,0.77]}

class1precision_w = {0.5:[0.91,0.92,0.92,0.91],
                     0.6:[0.90,0.90,0.91,0.91],
                     0.8:[0.89,0.89,0.89,0.90],
                     1.0:[0.88,0.87,0.88,0.89],
                     1.5:[0.85,0.83,0.83,0.85],
                     2.0:[0.81,0.81,0.80,0.80]}

class1recall_w =    {0.5:[0.79,0.79,0.78,0.86],
                     0.6:[0.82,0.79,0.81,0.80],
                     0.8:[0.82,0.82,0.83,0.81],
                     1.0:[0.86,0.84,0.84,0.84],
                     1.5:[0.93,0.92,0.93,0.89],
                     2.0:[0.96,0.96,0.96,0.94]}

fig1 = plt.figure(1)

ax = fig1.add_subplot(2,1,1)
class0pavg = []
for i in class0precision_w.keys():
   class0pavg.append(sum(class0precision_w[i])/4)
class0ravg = []
for i in class0recall_w.keys():
   class0ravg.append(sum(class0recall_w[i])/4)

ax.plot(c1weights,class0pavg,'r',label='non road precision')
ax.plot(c1weights,class0ravg,'g',label='non road recall')
ax.legend(loc=4)

ax = fig1.add_subplot(2,1,2)
class1pavg = []
for i in class1precision_w.keys():
   class1pavg.append(sum(class1precision_w[i])/4)
class1ravg = []
for i in class1recall_w.keys():
   class1ravg.append(sum(class1recall_w[i])/4)

ax.plot(c1weights,class1pavg,'r',label='road precision')
ax.plot(c1weights,class1ravg,'g',label='road recall')
ax.legend(loc=4)

fig1.text(0.5, 0.04, 'Weights on class 1 (road)',ha='center')
fig1.text(0.04, 0.5, 'Rate', va = 'center', rotation ='vertical')
plt.show()

#Penalty term#
cpen = [0.5,0.7,0.9,1.0,1.1,1.3,1.5]

class0precision_pen = {0.5:[0.86,0.87,0.86,0.86],
                       0.7:[0.86,0.88,0.85,0.86],
                       0.9:[0.85,0.87,0.85,0.86],
                       1.0:[0.85,0.86,0.86,0.86],
                       1.1:[0.88,0.86,0.87,0.86],
                       1.3:[0.85,0.87,0.86,0.88],
                       1.5:[0.86,0.88,0.86,0.87]}

class0recall_pen =    {0.5:[0.87,0.88,0.88,0.88],
                       0.7:[0.88,0.88,0.89,0.89],
                       0.9:[0.88,0.88,0.89,0.88],
                       1.0:[0.87,0.89,0.88,0.88],
                       1.1:[0.88,0.88,0.90,0.89],
                       1.3:[0.88,0.89,0.89,0.88],
                       1.5:[0.88,0.88,0.89,0.88]}

class1precision_pen = {0.5:[0.87,0.88,0.88,0.88],
                       0.7:[0.87,0.87,0.88,0.88],
                       0.9:[0.88,0.88,0.89,0.88],
                       1.0:[0.87,0.88,0.89,0.88],
                       1.1:[0.88,0.88,0.88,0.89],
                       1.3:[0.88,0.88,0.89,0.88],
                       1.5:[0.88,0.88,0.86,0.88]}

class1recall_pen =    {0.5:[0.86,0.87,0.86,0.86],
                       0.7:[0.86,0.87,0.84,0.86],
                       0.9:[0.84,0.87,0.85,0.85],
                       1.0:[0.85,0.85,0.87,0.86],
                       1.1:[0.88,0.86,0.86,0.86],
                       1.3:[0.85,0.87,0.85,0.88],
                       1.5:[0.86,0.88,0.86,0.87]}

fig2 = plt.figure(2)

ax = fig2.add_subplot(2,1,1)
class0pavg = []
for i in class0precision_pen.keys():
   class0pavg.append(sum(class0precision_pen[i])/4)
class0ravg = []
for i in class0recall_pen.keys():
   class0ravg.append(sum(class0recall_pen[i])/4)

ax.plot(cpen,class0pavg,'r',label='non road precision')
ax.plot(cpen,class0ravg,'g',label='non road recall')
ax.legend(loc='center right')

ax = fig2.add_subplot(2,1,2)
class1pavg = []
for i in class1precision_pen.keys():
   class1pavg.append(sum(class1precision_pen[i])/4)
class1ravg = []
for i in class1recall_pen.keys():
   class1ravg.append(sum(class1recall_pen[i])/4)

ax.plot(cpen,class1pavg,'r',label='road precision')
ax.plot(cpen,class1ravg,'g',label='road recall')
ax.legend(loc='center left')

fig1.text(0.5, 0.04, 'Penalty on error',ha='center')
fig1.text(0.04, 0.5, 'Rate', va = 'center', rotation ='vertical')
plt.show()


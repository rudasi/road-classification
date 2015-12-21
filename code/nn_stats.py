import numpy as np
import matplotlib.pyplot as plt
import math

#Units per layer#
units = [4,5,6,7,8,12]
class0precision_units =  {4:[0.86,0.84,0.85,0.85],
                          5:[0.89,0.87,0.88,0.88],
                          6:[0.88,0.89,0.89,0.88],
                          7:[0.89,0.87,0.89,0.88],
                          8:[0.90,0.88,0.89,0.87],
                         12:[0.91,0.87,0.88,0.91]}

class0recall_units =     {4:[0.81,0.81,0.80,0.79],
                          5:[0.87,0.85,0.88,0.85],
                          6:[0.87,0.84,0.84,0.87],
                          7:[0.87,0.85,0.86,0.86],
                          8:[0.85,0.84,0.86,0.86],
                         12:[0.86,0.88,0.86,0.87]}

class1precision_units =  {4:[0.81,0.81,0.81,0.81],
                          5:[0.88,0.85,0.88,0.86],
                          6:[0.88,0.85,0.87,0.87],
                          7:[0.87,0.85,0.86,0.87],
                          8:[0.86,0.85,0.87,0.87],
                         12:[0.88,0.86,0.87,0.87]}

class1recall_units =     {4:[0.87,0.85,0.86,0.86],
                          5:[0.89,0.87,0.88,0.89],
                          6:[0.89,0.89,0.89,0.88],
                          7:[0.89,0.87,0.89,0.88],
                          8:[0.91,0.89,0.90,0.88],
                         12:[0.91,0.87,0.88,0.91]}

fig1 = plt.figure(1)

ax = fig1.add_subplot(2,1,1)
class0pavg = []
for i in class0precision_units.keys():
   class0pavg.append(sum(class0precision_units[i])/4)
class0ravg = []
for i in class0recall_units.keys():
   class0ravg.append(sum(class0recall_units[i])/4)

ax.plot(units,class0pavg,'r',label='non road precision')
ax.plot(units,class0ravg,'g',label='non road recall')
ax.legend(loc=4)

ax = fig1.add_subplot(2,1,2)
class1pavg = []
for i in class1precision_units.keys():
   class1pavg.append(sum(class1precision_units[i])/4)
class1ravg = []
for i in class1recall_units.keys():
   class1ravg.append(sum(class1recall_units[i])/4)

ax.plot(units,class1pavg,'r',label='road precision')
ax.plot(units,class1ravg,'g',label='road recall')
ax.legend(loc=4)

fig1.text(0.5, 0.04, 'Number of units in single layer',ha='center')
fig1.text(0.04, 0.5, 'Rate', va = 'center', rotation ='vertical')
plt.show()

#alpha - regularization#
alpha = [0.0001,0.001,0.01,0.1]
#alpha = [math.log(10000*i) for i in alpha]

class0precision_alpha = {0.0001:[0.90,0.87,0.87,0.88],
                          0.001:[0.89,0.88,0.88,0.91],
                           0.01:[0.89,0.88,0.87,0.88],
                            0.1:[0.87,0.87,0.88,0.89]}

class0recall_alpha =    {0.0001:[0.87,0.85,0.85,0.86],
                          0.001:[0.88,0.86,0.87,0.87],
                           0.01:[0.87,0.86,0.86,0.86],
                            0.1:[0.88,0.86,0.86,0.87]}

class1precision_alpha = {0.0001:[0.87,0.85,0.85,0.86],
                          0.001:[0.89,0.86,0.87,0.88],
                           0.01:[0.88,0.85,0.87,0.87],
                            0.1:[0.88,0.86,0.86,0.87]}

class1recall_alpha =    {0.0001:[0.90,0.87,0.87,0.89],
                          0.001:[0.89,0.88,0.89,0.91],
                           0.01:[0.89,0.88,0.87,0.88],
                            0.1:[0.87,0.87,0.88,0.89]}

fig2 = plt.figure(2)

ax = fig2.add_subplot(2,1,1)
class0pavg = []
for i in class0precision_alpha.keys():
   class0pavg.append(sum(class0precision_alpha[i])/4)
class0ravg = []
for i in class0recall_alpha.keys():
   class0ravg.append(sum(class0recall_alpha[i])/4)

ax.plot(alpha,class0pavg,'r',label='non road precision')
ax.plot(alpha,class0ravg,'g',label='non road recall')
ax.legend()

ax = fig2.add_subplot(2,1,2)
class1pavg = []
for i in class1precision_alpha.keys():
   class1pavg.append(sum(class1precision_alpha[i])/4)
class1ravg = []
for i in class1recall_alpha.keys():
   class1ravg.append(sum(class1recall_alpha[i])/4)

ax.plot(alpha,class1pavg,'r',label='road precision')
ax.plot(alpha,class1ravg,'g',label='road recall')
ax.legend()

fig2.text(0.5, 0.04, 'Regularization parameter(alpha)',ha='center')
fig2.text(0.04, 0.5, 'Rate', va = 'center', rotation ='vertical')
plt.show()

#num iters#
iters = [25,50,100,150,200]

class0precision_iters =  {25:[0.86,0.85,0.86,0.83],
                          50:[0.87,0.87,0.89,0.86],
                         100:[0.86,0.86,0.87,0.89],
                         150:[0.87,0.88,0.89,0.89],
                         200:[0.87,0.87,0.87,0.89]}

class0recall_iters =     {25:[0.86,0.84,0.84,0.83],
                          50:[0.83,0.86,0.87,0.84],
                         100:[0.86,0.86,0.86,0.87],
                         150:[0.84,0.85,0.85,0.87],
                         200:[0.85,0.87,0.85,0.86]}

class1precision_iters =  {25:[0.86,0.84,0.85,0.83],
                          50:[0.84,0.86,0.87,0.84],
                         100:[0.86,0.86,0.86,0.87],
                         150:[0.85,0.86,0.86,0.87],
                         200:[0.85,0.86,0.85,0.86]}

class1recall_iters =     {25:[0.86,0.85,0.87,0.83],
                          50:[0.88,0.88,0.90,0.86],
                         100:[0.86,0.87,0.87,0.89],
                         150:[0.88,0.88,0.89,0.89],
                         200:[0.87,0.87,0.87,0.90]}

fig3 = plt.figure(3)

ax = fig3.add_subplot(2,1,1)
class0pavg = []
for i in class0precision_iters.keys():
   class0pavg.append(sum(class0precision_iters[i])/4)
class0ravg = []
for i in class0recall_iters.keys():
   class0ravg.append(sum(class0recall_iters[i])/4)

ax.plot(iters,class0pavg,'r',label='non road precision')
ax.plot(iters,class0ravg,'g',label='non road recall')
ax.legend(loc=4)

ax = fig3.add_subplot(2,1,2)
class1pavg = []
for i in class1precision_iters.keys():
   class1pavg.append(sum(class1precision_iters[i])/4)
class1ravg = []
for i in class1recall_iters.keys():
   class1ravg.append(sum(class1recall_iters[i])/4)

ax.plot(iters,class1pavg,'r',label='road precision')
ax.plot(iters,class1ravg,'g',label='road recall')
ax.legend(loc=4)

fig3.text(0.5, 0.04, 'Number of iterations till termination',ha='center')
fig3.text(0.04, 0.5, 'Rate', va = 'center', rotation ='vertical')
plt.show()

import matplotlib.pyplot as plt

data = {
      0.5:(0.91,0.98,0.82,0.49),
      0.6:(0.93,0.98,0.85,0.63),
      0.7:(0.93,0.97,0.79,0.63),
      0.8:(0.94,0.96,0.77,0.71),
      1.0:(0.95,0.95,0.75,0.73),
      1.2:(0.95,0.95,0.74,0.75),
      1.4:(0.95,0.94,0.72,0.76),
      1.6:(0.95,0.94,0.71,0.77),
      1.8:(0.96,0.93,0.70,0.77),
      2.0:(0.96,0.93,0.68,0.78),
      2.4:(0.96,0.92,0.67,0.79),
      2.6:(0.96,0.92,0.66,0.80),
      3.0:(0.96,0.92,0.65,0.80),
      3.5:(0.97,0.91,0.65,0.86),
      4.0:(0.97,0.91,0.63,0.87)}

w = sorted(data.keys())
print w
nroadp = [data[i][0] for i in w]
nroadr = [data[i][1] for i in w]
roadp = [data[i][2] for i in w]
roadr = [data[i][3] for i in w]

plt.figure(1)
plt.plot(w,nroadp,'r', label='non-road precision')
plt.plot(w,nroadr,'g', label='non-road recall')
plt.plot(w,roadp,'b', label='road precision')
plt.plot(w,roadr,'y', label='road recall')
plt.legend()

plt.show()

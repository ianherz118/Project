from PIL import Image
from pylab import *
from numpy import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#============ Lectura del archivo en formato de excel =======================#
# datos = pd.read_csv('C:/Users/PSC54195/Documents/MATLAB/mnist_train.csv', sep=',',header=None)
datos = pd.read_excel('C:/Users/PSC54195/Downloads/numeros.xlsx', sep = ',', header = None)
numero = np.asmatrix(datos.values[:,0])
target = numero[0]
plt.close('all')
plt.figure(0)
plt.imshow(np.reshape(numero[0,1:],[25,25]), cmap = 'gray')
plt.show()
#<=========================================================================>#
#<== Arquitectura neuronal con 64 neuronas de entrada y 10 neuronas de salida

w1 = np.random.rand(64, 625)
b1 = np.random.rand(64, 1)
plt.figure(1)
plt.imshow(np.reshape(w1[0,:],[25,25]), cmap='gray')


w2 = np.random.rand(10, 64)
b2 = np.random.rand(10, 1)

alpha = 0.01
pat = numero[0, 1:]

for j in range (5000):
    for i in range (1):
        a0 = pat.T
        n1 = w1.dot(a0) + b1
        a1 = 1 / (1 + np.exp(-n1))
        n2 = w2.dot(a1) + b2
        a2 = n2
        t = np.asmatrix([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).T
        e = t - a2
        s2 = (-2)*1*e
        s1 = (np.diagflat((1-np.array(a1))*np.array(a1)).dot(w2.T)).dot(s2)
        w2 = w2 - (alpha*s2*a1.T)
        b2 = b2 - (alpha*s2)
        w1 = w1 - (alpha*s1*a0.T)
        b1 = b1 - (alpha*s1)

plt.figure(2)
plt.imshow(np.reshape(w1[1,:],[25,25]), cmap='gray')
plt.show()

# x = arange(-2, 2, 0.1)
# y = 1 + sin(pi/4*x)
# sal2 = zeros(size(x))

# for i in range (size(x)):
#         a0 = x[i]
#         n1 = (w1*a0) + b1
#         a1 = 1 / (1 + exp(-n1))
#         n2 = (w2.T).dot(a1) + b2
#         sal2[i] = n2

# plt.plot(x, y, '--r', x, sal2, 'k')
# plt.axis([-2.5, 2.5, -0.5, 2.5])
# plt.show()


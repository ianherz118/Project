import numpy as np
import matplotlib.pyplot as plt
#============ Lectura del archivo en formato de excel =======================#
def purelin (n):
    a = 1*n
    return a

def sigmoide(n):
    a = 1 / (1 + np.exp(-n))
    return a
#<=========================================================================>#
#<========================== Primera Capa =================================>#
w1 = np.random.rand(2, 1)
b1 = np.random.rand(2, 1)
# w1 = np.asmatrix([[-0.27],[-0.41]])
# b1 = np.asmatrix([[-0.48],[-0.13]])
#<========================== Segunda Capa =================================>#
w2 = np.random.rand(1, 2)
b2 = np.random.rand(1, 1)
# w2 = np.asmatrix([0.09, -0.17])
# b2 = np.asmatrix([0.48])


alpha = 0.01
pat = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
target = 1 + np.sin((np.pi/4)*pat)
rr = sigmoide(pat)

for j in range (10000):
    for i in range (len(pat)):
        a0 = pat[i]
        n1 = w1.dot(a0) + b1
        a1 = sigmoide(n1)
        n2 = w2.dot(a1) + b2
        a2 = purelin(n2)
        # A2 = sigmoide(n2)
        e = target[i] - a2
        dF2 = 1   #<======= Derivada de la funcion de salida
        # DF2 = np.multiply( (1-A2), (A2))
        s2 = (-2)*dF2*e
        dF1 =  np.diagflat( np.multiply( (1-a1), (a1)) ) #<===== Derivada de la funcion de entrada
        s1  = (dF1.dot(w2.T)).dot(s2)
        w2 = w2 - (alpha*s2*a1.T)
        b2 = b2 - (alpha*s2)
        w1 = w1 - (alpha*s1*a0.T)
        b1 = b1 - (alpha*s1)

x = np.arange(-2, 2, 0.1)
y = 1 + np.sin((np.pi/4)*x)
sal2 = np.zeros(np.size(x))

for i in range (np.size(x)):
        a0 = x[i]
        n1 = w1.dot(a0) + b1
        a1 = sigmoide(n1)
        n2 = w2.dot(a1) + b2
        sal2[i] = purelin(n2)

plt.plot(x, y, '--r', x, sal2, 'k')
plt.axis([-2.5, 2.5, -0.5, 2.5])
plt.show()


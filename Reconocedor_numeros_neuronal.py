from skimage import io, data, color
import numpy as np
import matplotlib.pyplot as plt 
import random 
plt.close('all') #Cierra todas las ventanas.

numero = np.array(np.zeros([1,626]))

numero[0,0] = 5 #Determiando que el primer dato sea 5. (número elegido)

#Para la red neuronal
#para neurona 1
w1 = np.random.randn(65,625) #para la matriz de peso
p1 = np.random.randn(1,625) #para la polarización
#para neurona 2
w2 = np.random.randn(10,65) #para la matriz de peso
p2 = np.random.randn(1,65) #para la polarización

#Razón de aprendizaje
alfa = 0.01
#patrón de entrada
patron = numero[0, 1:] 

if (numero[0,0] == 0 ):
    target = [1,0,0,0,0,0,0,0,0,0]
elif (numero[0,0] == 1):
    target = [0,1,0,0,0,0,0,0,0,0]
elif (numero[0,0] == 2):
    target = [0,0,1,0,0,0,0,0,0,0]
elif (numero[0,0] == 3):
    target = [0,0,0,1,0,0,0,0,0,0]
elif (numero[0,0] == 4):
    target = [0,0,0,0,1,0,0,0,0,0]
elif (numero[0,0] == 5):
    target = [0,0,0,0,0,1,0,0,0,0]
elif (numero[0,0] == 6):
    target = [0,0,0,0,0,0,1,0,0,0]
elif (numero[0,0] == 7):
    target = [0,0,0,0,0,0,0,1,0,0]
elif (numero[0,0] == 8):
    target = [0,0,0,0,0,0,0,0,1,0]
elif (numero[0,0] == 9):
    target = [0,0,0,0,0,0,0,0,0,1]

#Entrenamieno
for entrena in range(1):
    a0 = patron
    n1 = w1*a0 + p1 #salida neurona 1
    a1 = 1/(1 + np.exp(-n1)) #Para una fucnión exponencial función de primer neurona.
    n2 = w2*a1 + p2 #Salida de neurona 2
    a2 = 1*n2 #Salida todal de neurona, con función lineal.
    error = target - a2
    
    
    
    
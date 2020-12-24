import numpy as np
import pyaudio 
import wave
import matplotlib.pyplot as plt
import winsound
from scipy.io import wavfile
import time
plt.close('all')
# Leer la señal
m1, cancion = wavfile.read('senal.wav')
t1 = np.arange(len(cancion)) / float(m1) #Calcular el tiempo de la grabación
cancion = cancion / (2.**15) #Normalización}
#Leer el ruido
m2, ruido = wavfile.read('ruido_lab.wav')
t2 = np.arange(len(ruido)) / float(m2) #Calcular el tiempo de la grabación
ruido = ruido / (2.**15) #Normalización
#Target
target = cancion +  ruido
#<========================= Diseñar red neuronal =============================>
# Definir arquitectura de red
w = np.array([-0.5, 0.5, 0.3])
b = -0.4
patron = np.zeros((3,1))
salida = np.zeros((len(ruido), 1))
alfa = 0.1
# Entrenamiento
for i in range(len(ruido)):
    if (i == 0):
        patron[0] = ruido[i]
    elif (i == 1):
        patron[0] = ruido[i]
        patron[1] = ruido[i - 1] # Primer delay
    else:
        patron[0] = ruido[i]
        patron[1] = ruido[i - 1]
        patron[2] = ruido[i - 2] # Segundo delay      
    a = np.dot(w,patron) + b # Salida de la neurona
    e = target[i] - a   
    salida[i] = e    
    #Actualizar pesos y polarizaciones
    w = w + (2*alfa*e*patron.T)
    b = b + (2*alfa*e)   
    
son_rec = salida * (2.**15) # Sonido recuperado reescalado a su valor original 
son_rec = np.array(son_rec, dtype = np.int16)
wavfile.write('filtro.wav', m1, son_rec)

winsound.PlaySound(r'filtro.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
time.sleep(30)

w2 = np.array([-0.5, 0.5, 0.3])
b2 = -0.4
patron2 = np.zeros((3,1))
a2 = np.zeros((len(ruido), 1))
salida2 = np.zeros((len(ruido), 1))
           
# Entrenamiento
for i in range(len(ruido)):
    if (i == 0):
        patron2[0] = ruido[i]
    elif (i == 1):
        patron2[0] = ruido[i]
        patron2[1] = ruido[i - 1] # Primer delay
    else:
        patron2[0] = ruido[i]
        patron2[1] = ruido[i - 1]
        patron2[2] = ruido[i - 2] # Segundo delay    
    a2[i] = np.dot(w2,patron2) + b2 # Salida de la neurona
    e2 = target[i] - a2[i]
    salida2[i] = e2   
    #Actualizar pesos y polarizaciones
    w2 = w2 + (2*alfa*e2*patron2.T)
    b2 = b2 + (2*alfa*e2)   
     
son_rec2 = salida2 * (2.**15) # Sonido recuperado reescalado a su valor original 
son_rec2 = np.array(son_rec2, dtype = np.int16)
wavfile.write('filtro.wav', m1, son_rec2)
winsound.PlaySound(r'filtro.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
time.sleep(30)

plt.figure(figsize = (10,4))
plt.subplot(3,2,3)
plt.plot(t1,salida)
plt.title('Salida de la neurona')
plt.xlim(t1[0],t1[-1])
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud de la señal')
plt.subplot(3,2,1)
plt.plot(t1,cancion)
plt.title('Cancion original')
plt.xlim(t1[0],t1[-1])
plt.subplot(3,2,5)
plt.plot(t1,cancion,t1,salida)
plt.title('Diferencia original y salida de la neurona')
plt.xlim(t1[0],t1[-1])
plt.subplot(3,2,4)
plt.plot(t1,salida2)
plt.title('Salida de la neurona')
plt.xlim(t1[0],t1[-1])
plt.subplot(3,2,2)
plt.plot(t1,cancion)
plt.title('Cancion original')
plt.xlim(t1[0],t1[-1])
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud de la señal')
plt.subplot(3,2,6)
plt.plot(t1,cancion,t1,salida2)
plt.title('Diferencia original y salida de la neurona')
plt.xlim(t1[0],t1[-1])
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud de la señal')
plt.show()



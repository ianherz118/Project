import numpy as np
import pyaudio 
import wave
import matplotlib.pyplot as plt
import winsound
from mpl_toolkits.mplot3d import Axes3D
import statistics as stat

#<=========================== Grabación ======================================>
paquete = 512 #Tamaño del paquete (512 muestras)
sample = pyaudio.paInt16
canales = 2 #Canales de la tarjeta de audio (Estereo)
fs = 8000 #Frecuencia de muestreo
segundos = 4 #Tiempo de audio
archivo = 'patron.wav' #Nombre del archivo

obj_audio = pyaudio.PyAudio() #Objeto de audio

input('Presiona una tecla')
print('Inicia grabación...')

#Grabación
streaming = obj_audio.open(format = sample, channels = canales, rate = fs, 
                           frames_per_buffer = paquete, input = True)
tramas = []
sonido = []

for i in range(0,int(fs / paquete * segundos)):
    datos = streaming.read(paquete)
    tramas.append(datos)
    sonido.append(np.frombuffer(datos, dtype = np.int16))

#Detener grabación
streaming.stop_stream()
streaming.close()

obj_audio.terminate() #Cerrar objeto de audio

print('Grabación finalizada')

#<========================== Escuchar grabación ==============================>
# Guardar audio en .wav (crudo)
wf = wave.open(archivo,'wb')
wf.setnchannels(canales)
wf.setsampwidth(obj_audio.get_sample_size(sample))
wf.setframerate(fs)
wf.writeframes(b''.join(tramas))
wf.close()
winsound.PlaySound(archivo, winsound.SND_FILENAME | winsound.SND_ASYNC) #Ejecutar audio

#<====================== Procesamiento de la señal ===========================>
plt.close('all')

aa = np.hstack(sonido) #Junta los stacks de sonido en una sola variable
plt.figure(1)
plt.title('Señal de audio grabada')
plt.plot(aa)

aa_norm = aa / np.max(np.abs(aa)) #Señal normalizada
plt.figure(2)
plt.plot(aa_norm)
plt.title('Señal de audio normalizada')

#Quitar las partes muertas
bin1 = np.where(np.abs(aa_norm) >= 0.1, 1, 0) #Comparación, valor a la salida que cumple, valor que no cumple
senal_1 = [] #Señal para hacer el corte

for i in range(0, len(aa_norm) - 399, 1): #Seccionando en ventanas de 10ms 
    #Evitar overlap
    senal_1.append(np.mean(bin1[i:i+400])) #Ventanas del 5% de la fs de la señal #Suavizado de la señal

plt.figure(2)
plt.plot(senal_1)   
plt.show()

bin2 = np.where(np.array(senal_1) >= 0.1, 1, 0)
senal_2 = [] #Señal para hacer el corte
for i in range(0, len(bin2) - 399, 1): #Seccionando en ventanas de 10ms 
    if bin2[i] == 1:
        senal_2.append(aa_norm[i]) #Ventanas del 5% de la fs de la señal #Suavizado de la señal

plt.figure(3)
plt.plot(senal_2)
plt.title('Información importante del audio')
plt.show()

#<=============================== Clasificador ===============================>
# Filtro de preénfasis (Mejora de agudos)
corre = np.zeros(len(senal_2)) 
corre[1:-1] = senal_2[0:-2] #Copia de la señal original desplazada a la izquierda
pre = corre - (0.95 * np.array(senal_2)) #Factor de preénfasis (mayor a 0.8) por señal desplazada

plt.figure(4)
plt.plot(pre)
plt.title('Filtro de preénfasis')
plt.show()

# Eliminar ruidos muy agudos (Opcional)
for i in range(0, len(pre), 1):
    if np.abs(np.float16(pre[i])) > 0.7:
        pre[i] = 0

# Corrimiento de la señal
ventana = 400 # Tamaño de la ventana
desplazamiento = 80 # Desplazamiento de la señal
Ima = [] #V Variable para guardar datos
k = 0

for i in range(0, len(pre) - (ventana - 1), desplazamiento):
    k += 1
    #ventana de hamming ayuda con problemas de discontinuidad
    Fourier = np.abs(np.fft.fft(pre[i:ventana - 1 + i] * np.hamming(ventana - 1))) # Aplicar Fourier por ventanas
    Ima.append(Fourier[0:200]) # Quedarnos solo con la parte positiva
    
# Graficar en 3D las ventanas de mi señal
# X->Tiempo
# Y->Espectro en Fourier
# Z->Magnitud
[X, Y] = np.mgrid[0:k, 0:200]
Z = np.array(Ima)
fig = plt.figure(6)
ax = fig.gca(projection = '3d')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Fourier')
ax.set_zlabel('Magnitud')
surf = ax.plot_surface(X, Y, Z, cmap = 'coolwarm', linewidth = 0)
plt.title('Espectro del audio')

#Encontrar patrones de clasificación
# comp_1 = np.max(np.array(Ima), axis = 0)
# comp_2 = np.mean(np.array(Ima), axis = 0)
patron = stat.mode(Ima) #(comp_1 + comp_2) / 2

plt.figure(5)
plt.plot(patron)
plt.title('Patron de entrada')
plt.show()

#<========================== Prueba clasificador ==============================>
# np.savetxt('nueve.csv', patron) #Guardar patrones de entrada

L0 = np.loadtxt('cero.csv')
L1 = np.loadtxt('uno.csv')
L2 = np.loadtxt('dos.csv')
L3 = np.loadtxt('tres.csv')
L4 = np.loadtxt('cuatro.csv')
L5 = np.loadtxt('cinco.csv')
L6 = np.loadtxt('seis.csv')
L7 = np.loadtxt('siete.csv')
L8 = np.loadtxt('ocho.csv')
L9 = np.loadtxt('nueve.csv')


print(np.corrcoef(patron,L0))
print(np.corrcoef(patron,L1))
print(np.corrcoef(patron,L2))
print(np.corrcoef(patron,L3))
print(np.corrcoef(patron,L4))
print(np.corrcoef(patron,L5))
print(np.corrcoef(patron,L6))
print(np.corrcoef(patron,L7))
print(np.corrcoef(patron,L8))
print(np.corrcoef(patron,L9))

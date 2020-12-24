import librosa 
import librosa.display 
import IPython.display as ipd
import numpy as np
import pyaudio 
import wave
import matplotlib.pyplot as plt
import winsound
from mpl_toolkits.mplot3d import Axes3D

paquete = 512 #Tamaño del paquete (512 muestras)
sample = pyaudio.paInt16
canales = 1 #Canales de la tarjeta de audio (Estereo)
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

#Encontrar patrones de clasificación
scale_file = "patron.wav"
ipd.Audio(scale_file)
scale, sr = librosa.load(scale_file)
filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
filter_banks.shape
plt.figure(figsize=(25, 10))
librosa.display.specshow(filter_banks, 
                         sr=sr, 
                         x_axis="linear")
plt.colorbar(format="%+2.f")
plt.show()
mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=10)
mel_spectrogram.shape
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
log_mel_spectrogram.shape
plt.figure(figsize=(25, 10))
librosa.display.specshow(log_mel_spectrogram, 
                         x_axis="time",
                         y_axis="mel", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()

#<========================== Prueba clasificador ==============================>
# 
# np.savetxt('Cero.csv', log_mel_spectrogram) #Guardar patrones de entrada

L1 = np.loadtxt('Uno.csv')
L2 = np.loadtxt('Dos.csv')
L3 = np.loadtxt('Tres.csv')
L4 = np.loadtxt('Cuatro.csv')
L5 = np.loadtxt('Cinco.csv')
L6 = np.loadtxt('Seis.csv')
L7 = np.loadtxt('Siete.csv')
L8 = np.loadtxt('Ocho.csv')
L9 = np.loadtxt('Nueve.csv')
L0 = np.loadtxt('Cero.csv')

c1=np.corrcoef(log_mel_spectrogram,L1)
c2=np.corrcoef(log_mel_spectrogram,L2)
c3=np.corrcoef(log_mel_spectrogram,L3)
c4=np.corrcoef(log_mel_spectrogram,L4)
c5=np.corrcoef(log_mel_spectrogram,L5)
c6=np.corrcoef(log_mel_spectrogram,L6)
c7=np.corrcoef(log_mel_spectrogram,L7)
c8=np.corrcoef(log_mel_spectrogram,L8)
c9=np.corrcoef(log_mel_spectrogram,L9)
c0=np.corrcoef(log_mel_spectrogram,L1)

a1=np.mean(c1)
a2=np.mean(c2)
a3=np.mean(c3)
a4=np.mean(c4)
a5=np.mean(c5)
a6=np.mean(c6)
a7=np.mean(c7)
a8=np.mean(c8)
a9=np.mean(c9)
a0=np.mean(c0)

print("Numero 1")
print(a1)
print("Numero 2")
print(a2)
print("Numero 3")
print(a3)
print("Numero 4")
print(a4)
print("Numero 5")
print(a5)
print("Numero 6")
print(a6)
print("Numero 7")
print(a7)
print("Numero 8")
print(a8)
print("Numero 9")
print(a9)
print("Numero 0")
print(a0)

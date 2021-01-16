from playsound import playsound
import time
import pyttsx3 as pyttsx
from gtts import gTTS
import os
import numpy as np
import pyaudio 
import wave
import matplotlib.pyplot as plt
import winsound
from detecta import detect_peaks

#Es importante que despues de correr el programa se borren los archivos wav generados
#Ya que al sobreescribir los datos marca error 

ind=[0,0]
wi=0
wiz=0
con=[]
ind=[0]
codeMorse=[]
count=0

dicc = { ' ':'/', 'A':'.-', 'B':'-...', 'C':'-.-.', 'D':'-..', 'E':'.', 'F':'..-.', 'G':'--.', 'H':'....',
                    'I':'..', 'J':'.---', 'K':'-.-', 'L':'.-..', 'M':'--', 'N':'-.','O':'---', 'P':'.--.', 'Q':'--.-',
                    'R':'.-.', 'S':'...', 'T':'-','U':'..-', 'V':'...-', 'W':'.--','X':'-..-', 'Y':'-.--', 'Z':'--..',
                    '1':'.----', '2':'..---', '3':'...--','4':'....-', '5':'.....', '6':'-....','7':'--...', '8':'---..', '9':'----.',
                    '0':'-----', ', ':'--..--', '.':'.-.-.-','?':'..--..', '/':'-..-.', '-':'-....-','(':'-.--.', ')':'-.--.-'}

def Txt_to_Morse():
    txt = input('Enter Text to Convert to Morse: ')
    code = [dicc[i.upper()] + ' ' for i in txt if i.upper() in dicc.keys()]
    morse=''.join(code)
    print(morse)
    for m in morse:
        if m=='.':
            playsound('dit.wav')
        elif m=='-':
            playsound('dah.wav')
        else:
            time.sleep(0.5)
            
def voz(text_file, lang, name_file):
    with open(text_file, "r") as file:
        text = file.read()
    file = gTTS(text=text,lang=lang)
    filename = name_file
    file.save(filename)
    
def Morse_to_Txt():
    wi=0
    print("Welcome to program Convert AudioCodeMorse-Audiotext")
    seconds=int(input("Duration of message:(seconds): "))
    while wi<=seconds: 
        paquete = 512 #Tamaño del paquete (512 muestras)
        sample = pyaudio.paInt16
        canales = 1 #Canales de la tarjeta de audio (Estereo)
        fs = 8000 #Frecuencia de muestreo
        segundos = 1 #Tiempo de audio
        archivo = 'morse.wav' #Nombre del archivo
        obj_audio = pyaudio.PyAudio() #Objeto de audio
        time.sleep(0.5)
        print('Obtain data...')
        streaming = obj_audio.open(format = sample, channels = canales, rate = fs, 
                              frames_per_buffer = paquete, input = True)
        tramas = []
        sonido = []
        for i in range(0,int(fs / paquete * segundos)):
            datos = streaming.read(paquete)
            tramas.append(datos)
            sonido.append(np.frombuffer(datos, dtype = np.int16))
        streaming.stop_stream()
        streaming.close()   
        obj_audio.terminate() #Cerrar objeto de audio   
        print('Processing')   
        #<========================== Escuchar grabación ==============================>
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
        aa_norm = aa / np.max(np.abs(aa)) #Señal normalizada
        #Quitar las partes muertas
        bina = np.where(np.abs(aa_norm) >= 0.1, 1, 0) #Comparación, valor a la salida que cumple, valor que no cumple
        senal_1 = [] #Señal para hacer el corte    
        senal_2 = [] #Señal para hacer el corte
        for i in range(0, len(aa_norm) - 399, 1): #Seccionando en ventanas de 10ms 
            if bina[i] == 1:
                senal_2.append(aa_norm[i]) #Ventanas del 5% de la fs de la señal #Suavizado de la señal
        # plt.figure()
        # plt.plot(senal_2)
        ind=detect_peaks(senal_2, mph=-0.8, mpd=120, show=True)
        n=len(ind)
        if ind[n-1]>2500:  
            con.append('-')
            # print("Silence")
        elif ind[n-1]>700 and ind[n-1]<2500:
            con.append(' ')
        else:
            con.append('.')
        print("Results:")
        m=len(con)
        print(con[m-1])
        codeMorse.append(con[m-1])
        mx=len(codeMorse)-1
        wi+=1
    
    codeConv = ''.join(codeMorse)
    codeConv=codeConv.replace("  ", "/")
    code= [k for i in codeConv.split() for k,v in dicc.items() if i==v]
    
    newtxt = ''.join(code)
    
    print("Original message(Morse)")
    print(codeMorse)
    print(codeConv)#Se separa las palabras con doble espacio "  "= /
    
    for m in codeMorse:
            if m=='.':
                playsound('dit.wav')
            elif m=='-':
                playsound('dah.wav')
            else:
                time.sleep(0.5)
            
    with open("contenido2.txt", "w") as file:
        file.write(codeConv)
        file.write("\n")
        file.write("Fue el codigo Morse registrado")
        file.close()
    
    time.sleep(1)
    
    voz("contenido2.txt","ES","MorseCode.mp3")
    print("Reproduciendo:")
    audio ="MorseCode.mp3";
    playsound(audio)
    print("Reproducido.")
    
    print("Converted message ")
    print(newtxt)
    
    
    with open("contenido.txt", "w") as file:
        file.write(newtxt)
        file.write("\n")
        file.write("Fue la palabra o fase convertida del codigo registrado")
        file.close()
            
    voz("contenido.txt","ES","voz.mp3")
    print("Reproduciendo:")
    audio ="voz.mp3";
    playsound(audio)
    print("Reproducido.")
    
    print("Fin de la conversion del programa")

print('''\n1 - Convert Text to Morse \n2 - Convert Morse to Text\n3 - Quit\n ''')

while True:
        selection = int(input('Select Your Choice: '))
        if selection == 1:
            print(Txt_to_Morse())
            break
        elif selection == 2:
            print(Morse_to_Txt())
            break
        elif selection == 3:
            print('Exiting')
            break
        else:
            print('Wrong Selection, enter again')
 






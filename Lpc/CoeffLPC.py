#calculate short time fourier transform and plot spectrogram 
# from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

#================= calculate LPC coefficients from sound file ================
def autocorr(x):
    n = len(x)
    variance = np.var(x)
    x = x - np.mean(x)
    #n numbers from last index
    r = np.correlate(x, x, mode = 'full')[-n:]
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

def createSymmetricMatrix(acf,p):
    R = np.empty((p,p))
    for i in range(p):
        for j in range(p):
            R[i,j] = acf[np.abs(i-j)]
    return R

def lpc(s,fs,p):
    
    #divide into segments of 25 ms with overlap of 10ms
    nSamples = np.int32(0.025*fs)
    overlap = np.int32(0.01*fs)
    nFrames = np.int32(np.ceil(len(s)/(nSamples-overlap)))
    
    #zero padding to make signal length long enough to have nFrames
    padding = ((nSamples-overlap)*nFrames) - len(s)
    if padding > 0:
        signal = np.append(s, np.zeros(padding))
    else:
        signal = s
    segment = np.empty((nSamples, nFrames))
    start = 0
    for i in range(nFrames):
        segment[:,i] = signal[start:start+nSamples]
        start = (nSamples-overlap)*i
    
    #calculate LPC with Yule-Walker
    lpc_coeffs = np.empty((p, nFrames))
    for i in range(nFrames):
        acf = autocorr(segment[:,i])
        r = -acf[1:p+1].T
        R = createSymmetricMatrix(acf,p)
        lpc_coeffs[:,i] = np.dot(np.linalg.inv(R),r)
        lpc_coeffs[:,i] = lpc_coeffs[:,i]/np.max(np.abs(lpc_coeffs[:,i]))
    
    return lpc_coeffs

#calculate Euclidean distance between two matrices
def EUDistance(d,c):

    # np.shape(d)[0] = np.shape(c)[0]
    n = np.shape(d)[1]
    p = np.shape(c)[1]
    distance = np.empty((n,p))

    if n<p:
        for i in range(n):
            copies = np.transpose(np.tile(d[:,i], (p,1)))
            distance[i,:] = np.sum((copies - c)**2,0)
    else:
        for i in range(p):
            copies = np.transpose(np.tile(c[:,i],(n,1)))
            distance[:,i] = np.transpose(np.sum((d - copies)**2,0))

    distance = np.sqrt(distance)
    return distance

def lbg(features, M):
    eps = 0.0001
    codebook = np.mean(features, 1)
    distortion = 1
    nCentroid = 1
    while nCentroid < M:
        
        #double the size of codebook
        new_codebook = np.empty((len(codebook), nCentroid*2))
        if nCentroid == 1:
            new_codebook[:,0] = codebook*(1+eps)
            new_codebook[:,1] = codebook*(1-eps)
        else:
            for i in range(nCentroid):
                new_codebook[:,2*i] = codebook[:,i] * (1+eps)
                new_codebook[:,2*i+1] = codebook[:,i] * (1-eps)
                
        codebook = new_codebook
        nCentroid = np.shape(codebook)[1]
        D = EUDistance(features, codebook)
        
        while np.abs(distortion) > eps:
            #nearest neighbour search
            prev_distance = np.mean(D)
            nearest_codebook = np.argmin(D,axis = 1)
            
            #cluster vectors and find new centroid
            for i in range(nCentroid):
                #add along 3rd dimension
                codebook[:,i] = np.mean(features[:,np.where(nearest_codebook == i)], 2).T
            
            #replace all NaN values with 0
            codebook = np.nan_to_num(codebook)
            D = EUDistance(features, codebook)
            distortion = (prev_distance - np.mean(D))/prev_distance
            #print 'distortion' , distortion
    #print 'final codebook', codebook, np.shape(codebook)
    return codebook

def training(orderLPC):
    nSpeaker = 10
    nCentroid = 16
    codebooks_lpc = np.empty((nSpeaker, orderLPC, nCentroid))
    directory = 'C:/Users/giova/Desktop/lpc/test'
    # directory = 'C:/Users/PSC54195/Documents/Reconocimiento_Patrones/Parcial_2_Python/Files/test'
    fname = str()
    
    for i in range(nSpeaker):
        # fname = '/s' + str(i) + '.wav'
        fname = '/Num' + str(i) + '.wav'
        print ('Now speaker ', str(i), 'features are being trained')
        fs, s = wavfile.read(directory + fname)
        lpc_coeff = lpc(s, fs, orderLPC)
        codebooks_lpc[i,:,:] = lbg(lpc_coeff, nCentroid)
        plt.figure(i)
        plt.title('Codebook for speaker ' + str(i) + ' with ' + str(nCentroid) + ' centroids')
        
        for j in range(nCentroid):
            markerline, stemlines, baseline = plt.stem(codebooks_lpc[i,:,j])
            plt.setp(markerline,'markerfacecolor','r')
            plt.setp(baseline,'color', 'k')
            plt.ylabel('LPC')
            plt.axis(ymin = -1, ymax = 1)
            plt.xlabel('Number of features')
            
    plt.show()
    print ('Training complete')

    
    for i in range(2):
        # fname = '/s' + str(i) + '.wav'
        fname = '/Num' + str(i) + '.wav'
        fs,s = wavfile.read(directory + fname)
    
    return (codebooks_lpc)


def minDistance(features, codebooks):
    speaker = 0
    distmin = np.inf
    for k in range(np.shape(codebooks)[0]):
        D = EUDistance(features, codebooks[k,:,:])
        dist = np.sum(np.min(D, axis = 1))/(np.shape(D)[0])
        if dist < distmin:
            distmin = dist
            speaker = k
    return speaker

nSpeaker = 10
nfiltbank = 12
orderLPC = 15
codebooks_lpc = training(orderLPC)
directory = 'C:/Users/giova/Desktop/lpc/train'
# directory = 'C:/Users/PSC54195/Documents/Reconocimiento_Patrones/Parcial_2_Python/Files/train'
fname = str()
nCorrect_LPC = 0

for i in range(nSpeaker):
    # fname = '/s' + str(i) + '.wav'
    fname = '/Num' + str(i) + '.wav'
    print ('Now speaker ', str(i), 'features are being tested')
    fs,s = wavfile.read(directory + fname)
    lpc_coefs = lpc(s, fs, orderLPC)
    sp_lpc = minDistance(lpc_coefs, codebooks_lpc)
    
    print ('Speaker', (i), ' in test matches with speaker ', (sp_lpc), 'in train for training with LPC')
    
    if i == sp_lpc:
        nCorrect_LPC += 1

percentageCorrect_LPC = (nCorrect_LPC/nSpeaker)*100
print ('Accuracy of result for training with LPC is ', percentageCorrect_LPC, '%')







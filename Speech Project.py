# copyright - Parsa Mazaheri 

import sys
import os
import subprocess

#install packages automaticlly if they're not installed
for package in ['numpy', 'scipy', 'matplotlib']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
print("\n ---------------------------------------------------------------------------- \n")

# import packages
import numpy as np
from scipy.signal import find_peaks
from scipy.io import wavfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('bmh')



# get the current path
path = os.path.dirname(os.path.realpath(__file__))


# initialize plots
fig = plt.figure(constrained_layout=True, figsize=(10, 5))
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
f_ax1 = fig.add_subplot(spec[0, 0])
f_ax2 = fig.add_subplot(spec[0, 1])
f_ax3 = fig.add_subplot(spec[1, 0])
f_ax4 = fig.add_subplot(spec[1, 1])


# read the audio
filename = 'a.wav'
FS, y = wavfile.read(path + '/vowels/' + filename)

frame_size = 400
time_vector = np.arange(frame_size) / FS

# select frame
index = 6800
frame = y[index : index + frame_size]

# add the wave to first plot
f_ax1.plot(time_vector, frame)
f_ax1.set_xlabel('time (s)')
f_ax1.set_title('Signal Time frame')
# show the selected frame on plot
f_ax1.text(0.2, 0.2, "Frame: [{} : {}]".format(index, index+frame_size), horizontalalignment='center', 
            verticalalignment='center', transform=f_ax1.transAxes, color='red')


## windowing 
win = np.hamming(frame_size)
windowed_frame = win * frame

## normalize
windowed_frame -= windowed_frame.max()

## Fourier space 
# Calculating  log(abs(fft(x)))
dt = 1 / FS
freq_vector = np.fft.rfftfreq(frame_size, d=dt)
X = np.fft.rfft(windowed_frame)
log_X = np.log(np.abs(X))

# plot log(abs(fft(x)))
f_ax2.plot(log_X)
f_ax2.set_xlabel('frequency (Hz)')
f_ax2.set_title('Fourier spectrum')


## cepstrum
cepstrum = np.fft.rfft(log_X)
df = freq_vector[1] - freq_vector[0]
quefrency_vector = np.fft.rfftfreq(log_X.size, df)


# high liftering function
def high_liftering(c):
    N = 20
    for i in range(N):
        c[i] = 0
    return c

# low liftering function
def low_liftering(c):
    N = 20
    for i in range(N, len(c)):
        c[i] = 0
    return c


## Pitch
def calculate_pitch():
    cepstrum_pitch = high_liftering(np.copy(cepstrum))

    # pitch detection
    max_quefrency_index = np.argmax(np.abs(cepstrum_pitch))
    print("Max index: ", max_quefrency_index)
    pitch_freq = 1/quefrency_vector[max_quefrency_index]

    # plot
    f_ax3.plot(quefrency_vector, np.abs(cepstrum_pitch))
    f_ax3.set_xlabel('quefrency (s)')
    f_ax3.set_title('Cepstrum')
    f_ax3.plot(quefrency_vector[max_quefrency_index], np.abs(cepstrum_pitch)[max_quefrency_index], "o")
    # show pitch on plot
    f_ax3.text(0.8, 0.9, "Pitch: {0:.2f} Hz".format(pitch_freq), horizontalalignment='center', 
                verticalalignment='center', transform=f_ax3.transAxes, color='red')
    
    print("Pitch: ", pitch_freq)


## Formant
def calculate_formant():
    cepstrum_formant = low_liftering(np.copy(cepstrum))

    # get another fft of the cepstrum => log(abs(fft_result))
    formant = np.fft.rfft(cepstrum_formant)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector2 = np.fft.rfftfreq(formant.size, df)
    formant_X = np.fft.rfft(formant)
    log_formant_X = np.log(np.abs(formant_X))

    # plot
    f_ax4.plot(quefrency_vector2, np.abs(log_formant_X))
    f_ax4.set_xlabel('quefrency (s)')
    f_ax4.set_title('Cepstrum formant')
    
    # find peaks 
    peaks, _ = find_peaks(log_formant_X)
    f_ax4.plot(quefrency_vector2[peaks], log_formant_X[peaks], "o")
    # show formant on plot
    fp = ["{0:.3f}".format(item) for item in log_formant_X[peaks]]
    f_ax4.text(0.6, 0.9, fp, horizontalalignment='center', 
                verticalalignment='center', transform=f_ax4.transAxes, color='red')

    print("Formant: ", log_formant_X[peaks])
    

# calc pitch
calculate_pitch()

# calc formant
calculate_formant()

# show plots
plt.show()
#plt.savefig("plots/" + filename.split(".")[0] + ".png")
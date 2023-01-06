import os
import matplotlib.pyplot as plt
import vallenae as vae
import pywt
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, rfft

# Read waveform from tradb
HERE = "D:\Thesis\\acoustics\Data collected\9-11-2022\continous_mode_with_lens"
TRADB = os.path.join(HERE, "multiple_scratch_CM.tradb")  # uncompressed
TRAI = 17
fs = 2e6

with vae.io.TraDatabase(TRADB) as tradb:
    y0, t0 = tradb.read_wave(TRAI)

x = y0 * 1e3  # in mV
t = t0 * 1e6  # for µs

wave_type = "db2"
level = 3
coeffs = pywt.wavedec(x, wave_type, level=level)

cA3, cD3, cD2, cD1 = coeffs
print(len(cA3), len(cD3), len(cD2), len(cD1))


# Calculate standard deviations of each coefficient
cA3_std = np.std(cA3)
cD3_std = np.std(cD3)
cD2_std = np.std(cD2)
cD1_std = np.std(cD1)
print(cD3_std)

# Choose a threshold factor
threshold_factor = 4

# Calculate threshold values for each coefficient
cA3_threshold = cA3_std * threshold_factor
cD3_threshold = cD3_std * threshold_factor
cD2_threshold = cD2_std * threshold_factor
cD1_threshold = cD1_std * threshold_factor

# Set values below the threshold to zero
czD3 = cD3 * (cD3 > cD3_threshold)
print(czD3)
czD2 = cD2 * (cD2 > cD2_threshold)
czD1 = cD1 * (cD1 > cD1_threshold)

"""""
czD3 = cD3 * 0
czD2 = cD2 * 0
czD1 = cD1 * 0
"""
# Reconstruct the filtered signal using the modified coefficients
new_coeff = [cA3,czD3, czD2, czD1]
xw = pywt.waverec(new_coeff, wave_type)

plt.figure(figsize=(6, 12))
plt.subplot(411)
plt.plot(cA3)
plt.ylabel("cA3")
plt.subplot(412)
plt.plot(cD3)
plt.ylabel("cD3")
plt.subplot(413)
plt.plot(cD2)
plt.ylabel("cD2")
plt.subplot(414)
plt.plot(cD1)
plt.ylabel("cD1")
plt.show()

plt.figure()
plt.subplot(211)
plt.plot(t,x)
plt.ylabel("Orginal signal")
plt.subplot(212)
plt.plot(t,xw)
plt.ylabel("filtered signal")
plt.show()

def HFFT(x):
    X = fft(x)
    N = len(X)  # len(X) = len(t0)
    # to recover accurate units (eg:mV)
    X_mag = np.abs(X) /N    # step1: divide fourier coefficients by N
    pos = (N//2)
    X_mag = 2 * X_mag[:pos]  # step2: double the positive freq (negative freq we avoid)
    X_mag[0] = X_mag[0]/2   #0Hz and nyquist freq will not have negative side so no need to multiply by 2

    fs = 2000000 #sampling rate/freq
    ts = 1/fs    #sampling period
    freq0 = (fftfreq(N, ts))/1000  #KHz
    freq = freq0[:pos]

    #concept from utube channel "Mike X cohen" the output of fftfreq and below method are same.
    #freq2 = np.linspace(0,fs/2,(len(t0)//2)+1)
    #print("freq2",freq2)

    # max amplitude and coreesponding freq
    X_mag_sorted = X_mag
        # sort the array in descending order
    X_mag_sorted = sorted(X_mag_sorted, reverse=True)
        # return the element at the first position of the sorted array
    X_max = X_mag_sorted[0]
    index = np.where(X_mag == X_max)
    freq_of_max_ampl = (round((float(freq[index])), 2))

    X_2max = X_mag_sorted[1]
    index = np.where(X_mag == X_2max)
    freq_of_2ndmax_ampl = (round((float(freq[index])), 2))

    return freq,X_mag,freq_of_max_ampl,freq_of_2ndmax_ampl

if __name__ == "__main__":

    freq,X_mag,freq_of_max_ampl,freq_of_2ndmax_ampl = HFFT(x)
    plt.figure(figsize=(10, 6))
    plt.subplot(121)
    plt.plot(freq, X_mag)
    plt.xlabel("Frequency in KHz")
    plt.ylabel("FFT amplitude")
    plt.title(f'orginal signal Frequency plot, Highest peak frequency = {freq_of_max_ampl}KHz,\n 2ndmax = {freq_of_2ndmax_ampl}KHz')


    freq,X_mag,freq_of_max_ampl,freq_of_2ndmax_ampl = HFFT(xw)
    plt.subplot(122)
    plt.plot(freq, X_mag)
    plt.xlabel("Frequency in KHz")
    plt.ylabel("FFT amplitude")
    plt.title(f'Filtered Frequency plot, Highest peak frequency = {freq_of_max_ampl}KHz, \n 2ndmax = {freq_of_2ndmax_ampl}KHz')
    plt.show()

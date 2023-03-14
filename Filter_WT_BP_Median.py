"""
Read and plot transient data
============================
"""

import os
import matplotlib.pyplot as plt
import vallenae as vae
from scipy.fftpack import fft, fftfreq
import numpy as np
import pywt
import scipy


pi = np.pi

# Will take the path of the database from local path
HERE = "D:\Thesis\\acoustics\Data collected\\10-01-23"
TRADB = os.path.join(HERE, "with_lens_15_min.tradb")

TRAI = 67   #Transient record index  or window number = 10000 µs


fs = 2e6   # sampling frequency
N_tap = 121
f1, f2 = 35e3, 40e3     # bandpass filter frequency range

def main():
    # Read waveform from tradb
    with vae.io.TraDatabase(TRADB) as tradb:
        y0, t0 = tradb.read_wave(TRAI)

    x1 = y0 * 1e3  # in mV
    t = t0 * 1e6  # for µs

    x2 = WT(x1)
    taps = fir2(f1, f2, fs, N_tap)
    x3 = scipy.signal.lfilter(taps, 1, x2)
    x4 = median(x3)


    plt.figure()
    plt.subplot(311)
    plt.plot(t, x1)
    plt.title("Raw signal")
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude (mV)")

    plt.subplot(312)
    plt.plot(t, x2)
    plt.title("Wavelet filtered signal")
    plt.subplots_adjust(hspace=0.7)
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude (mV)")

    plt.subplot(313)
    plt.plot(t, x4)
    plt.title("Bandpass and median filtered signal")

    # Add some space between the subplots
    plt.subplots_adjust(hspace=0.7)
    plt.ylabel("Amplitude (mV)")

    plt.xlabel("Time (µs)")
    plt.show()

    return x3

#Bandpass filter
def fir2(f1, f2, fs, N_tap):
    taps = scipy.signal.firwin(N_tap, [f1, f2], pass_zero=False, fs = fs)
    return taps

#Median filter
def median(x3):
    x4 = scipy.signal.medfilt(x3, kernel_size=None)
    return x4

#Wavelet filter
def WT(x1):
    wave_type = "db16"
    level = 3
    coeffs = pywt.wavedec(x1, wave_type, level=level)

    cA3, cD3, cD2, cD1 = coeffs

    # Calculate standard deviations of each coefficient
    cA3_std = np.std(cA3)
    cD3_std = np.std(cD3)
    cD2_std = np.std(cD2)
    cD1_std = np.std(cD1)

    # Choose a threshold factor
    threshold_factor = 0.1

    # Calculate threshold values for each coefficient
    cA3_threshold = cA3_std * threshold_factor
    cD3_threshold = cD3_std * threshold_factor
    cD2_threshold = cD2_std * threshold_factor
    cD1_threshold = cD1_std * threshold_factor

    # Set values below the threshold to zero
    czA3 = cA3 * (cA3 > cA3_threshold)
    czD3 = cD3 * (cD3 > cD3_threshold)
    czD2 = cD2 * (cD2 > cD2_threshold)
    czD1 = cD1 * (cD1 > cD1_threshold)

    # Reconstruct the filtered signal using the modified coefficients
    new_coeff = [czA3, czD3, czD2, czD1]
    x2 = pywt.waverec(new_coeff, wave_type)

    return x2


#Fast fourier transform (FFT)
def HFFT(x):
    X = fft(x)
    N = len(X)  # len(X) = len(t0)
    # to recover accurate units (eg:mV)
    X_mag = np.abs(X) /N    # step1: divide fourier coefficients by N
    pos = (N//2)
    X_mag = 2 * X_mag[:pos]  # step2: double the positive freq (negative freq we avoid)
    X_mag[0] = X_mag[0]/2   #0Hz and nyquist freq will not have negative side so no need to multiply by 2

    fs = 2e6 #sampling rate/freq
    ts = 1/fs    #sampling period
    freq0 = (fftfreq(N, ts))/1000  #KHz
    freq = freq0[:pos]

    # max amplitude and coreesponding freq
    X_max = max(X_mag)
    index = np.where(X_mag == X_max)
    freq_of_max_ampl = (round((float(freq[index])), 2))

    plt.stem(freq, X_mag, 'b', \
             markerfmt=" ", basefmt="-b")  # 17 // 3  # floor division discards the fractional part
    plt.xlabel('Freq (KHz)')
    plt.ylabel('FFT Amplitude (mV)')
    plt.title(f'Highest peak frequency = {freq_of_max_ampl}KHz')
    plt.xlim(10, 60)
    plt.show()


if __name__ == "__main__":
    x = main()
    HFFT(x)


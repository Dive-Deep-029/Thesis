"""
Read and plot Unfiltered transient data, FFT
============================
"""

import os
import matplotlib.pyplot as plt
import vallenae as vae
from scipy.fftpack import fft, fftfreq
import numpy as np

pi = np.pi

# Local file path for TRADB database
HERE = "D:\Thesis\\acoustics\Data collected\\10-01-23"
TRADB = os.path.join(HERE, "with_lens_15_min.tradb")
TRAI = 67 # window number = 10000 µs
fs = 2e6


def main():
    # Read waveform from tradb
    with vae.io.TraDatabase(TRADB) as tradb:
        y0, t0 = tradb.read_wave(TRAI)

    x = y0 * 1e3  # in mV
    t = t0 * 1e6  #  µs

    return x,t

def HFFT(x,t):
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

    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.plot(t, x)
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"Unfiltered Transient Wave Plot")


    # max amplitude and coreesponding freq
    X_max = max(X_mag)
    index = np.where(X_mag == X_max)
    freq_of_max_ampl = (round((float(freq[index])), 2))

    plt.subplot(122)
    plt.stem(freq, X_mag, 'b', \
             markerfmt=" ", basefmt="-b")  # 17 // 3  # floor division discards the fractional part
    plt.xlabel('Freq (KHz)')
    plt.ylabel('FFT Amplitude (mV)')
    plt.title(f'Highest peak frequency = {freq_of_max_ampl}KHz')
    # plt.xlim(0, 400)
    plt.show()


if __name__ == "__main__":
    x,t = main()

    HFFT(x,t)


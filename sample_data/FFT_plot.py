"""
Read and plot transient data
============================
"""

import os

import matplotlib.pyplot as plt

import vallenae as vae

from scipy.fftpack import fft, ifft, fftfreq, rfft
import scipy
import numpy as np
import FFT_test_new
import filter
pi = np.pi
import pywt

from mpl_toolkits.mplot3d import Axes3D

#HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
HERE = "D:\Thesis\\acoustics\Data collected\9-11-2022\Wo_continous_mode_with_lens"
TRADB = os.path.join(HERE, "one_scratch4.tradb")  # uncompressed
TRAI = 1
fs = 2000000


def main():
    # Read waveform from tradb
    #print(type(TRADB))
    with vae.io.TraDatabase(TRADB) as tradb:
        y0, t0 = tradb.read_wave(TRAI)

    x = y0 * 1e3  # in mV
    t = t0 * 1e6  # for µs

    energy = vae.features.energy(x, 2000000)
    print("Energy",energy)
    # Plot waveforms
    #plt.figure(figsize=(8, 4), tight_layout=True)
    plt.plot(t, x)
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"Transient Wave Plot; trai = {TRAI}")
    #plt.show()
    return x,t

def HFFT(x,t):
    X = fft(x)
    print (X)
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
    #print("freq", freq[3])

    #concept from utube channel "Mike X cohen" the output of fftfreq and below method are same.
    #freq2 = np.linspace(0,fs/2,(len(t0)//2)+1)
    #print("freq2",freq2[3])<

    plt.figure(figsize=(16, 8))
    plt.subplot(221)
    # plt.figure(figsize=(8, 4), tight_layout=True)
    plt.plot(t, x)
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"Transient Wave Plot; trai = {TRAI}")

    plt.subplot(222)
        # x[start:end:step] default values  x[0:len(x):1]      # N//2 is to cancel negative freq
    plt.stem(freq, X_mag, 'b', \
             markerfmt=" ", basefmt="-b")  # 17 // 3  # floor division discards the fractional part
    # plt.plot(freq, np.abs(X))
    plt.xlabel('Freq (KHz)')
    plt.ylabel('FFT Amplitude (mV)')
    plt.title('FFT')
    # plt.xlim(0, 400)

    # PSD
    plt.subplot(223)
    psd = (X * np.conj(X)) / N
    plt.plot(freq, psd[:pos])
    plt.xlabel("Feq (KHz)")
    plt.ylabel("PSD")
    plt.xlim(0, 400)
    plt.title('PSD')
    #plt.tight_layout()
    #plt.show()

    # max amplitude and coreesponding freq
    X_max = max(X_mag)
    index = np.where(X_mag == X_max)
    freq_of_max_ampl = (round((float(freq[index])), 2))

    #filtering
    indices = psd > 0.001
    psd_clean = psd * indices

    plt.subplot(224)
    plt.plot(freq, psd_clean[:pos])
    plt.xlabel("Feq (KHz)")
    plt.ylabel("Filtered_PSD")
    plt.locator_params(axis='x', nbins=25)
    plt.xlim(0,300)
    plt.title(f'Filtered Frequency plot, Highest peak frequency = {freq_of_max_ampl}KHz')
    plt.tight_layout()
    plt.show()

def fir2(x, f1, f2, fs, N_tap=51):
    taps = scipy.signal.firwin(N_tap, [f1, f2], pass_zero=False, fs=fs)

    w, h = scipy.signal.freqz(taps, worN=8000)
    plt.figure()
    plt.plot((w / pi) * fs / 2, np.abs(h), linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency Response')
    # plt.ylim(-0.05, 1.05)
    plt.grid(True)

    x_filt = scipy.signal.lfilter(taps, 1.0, x)
    return x_filt

if __name__ == "__main__":
    x,t = main()

    HFFT(x,t)

    x_filt = fir2(x, 1e4, 20e4, fs, N_tap=51)

    plt.figure()
    plt.subplot(211)
    plt.plot(t, x, 'r')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(t, x_filt, 'r')
    plt.xlabel('Time')
    plt.ylabel('Amplitude_filtered')
    plt.grid(True)
    plt.show()

    HFFT(x_filt,t)

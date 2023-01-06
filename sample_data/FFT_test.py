import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
import scipy
import filter

#plt.style.use('seaborn-poster')


def fft_(x):
    X = fft(x)
    N = len(X)
    X_mag = np.abs(X) /N    # step1: divide fourier coefficients by N
    X_mag = 2 * X_mag[:(N//2)]  # step2: double the positive freq (negative freq we avoid)
    X_mag[0] = X_mag[0]/2

    fs = 2000 #sampling rate/freq
    ts = 1/fs    #sampling period
    freq = (fftfreq(N, ts))  #KHz
    freq = freq[:(N//2)]

    plt.figure(figsize = (12, 6))
    plt.subplot(121)

    plt.stem(freq, X_mag, 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 10)

    plt.show()

if __name__ == "__main__" :
    # sampling rate
    fs = 2000
    # sampling interval
    ts = 1.0/fs
    t = np.arange(0,1,ts)

    freq = 100.
    x = 3*np.sin(2*np.pi*freq*t)

    freq = 400
    x += np.sin(2*np.pi*freq*t)

    freq = 700
    x += 0.5* np.sin(2*np.pi*freq*t)

    #x = x + np.random.rand(2000)

    plt.plot(t, x, 'r')
    plt.ylabel('Amplitude')
    plt.show()


    def fir2(x, t, fs):
        # sampling freq
        nyqt = fs * 0.5
        f1, f2 = 200 / fs, 600 / fs  # normalizing freq
        N_tap = 51
        h = scipy.signal.firwin(N_tap, cutoff=[f1, f2], pass_zero=False)
        # n = np.arange(10000)
        # x_syn = np.sin(2 * np.pi * 50e3 * n / fs) + np.sin(2 * np.pi * 250e3 * n / fs)

        y_filt = scipy.signal.lfilter(h, [1.0], x)

        plt.figure()
        plt.subplot(211)
        plt.plot(t[:100], x[:100])
        plt.subplot(212)
        plt.plot(t[:100], y_filt[:100])
        plt.show()

        plt.figure()
        fft_(y_filt, fs)

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
import scipy

pi = np.pi

def fir2(x, f1, f2, fs, N_tap=51):
    taps = scipy.signal.firwin(N_tap, [f1, f2], pass_zero=False, fs = fs)

    w, h = scipy.signal.freqz(taps, worN=8000)
    plt.figure()
    plt.plot((w / pi) * fs/2, np.abs(h), linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency Response')
    #plt.ylim(-0.05, 1.05)
    plt.grid(True)


    x_filt = scipy.signal.lfilter(taps, 1.0, x)
    return x_filt


def mfft(x, fs):
    X = fft(x)
    N = len(X)
    X_mag = np.abs(X) /N    # step1: divide fourier coefficients by N
    X_mag = 2 * X_mag[:(N//2)]  # step2: double the positive freq (negative freq we avoid)
    X_mag[0] = X_mag[0]/2

    ts = 1/fs    #sampling period
    freq = (fftfreq(N, ts))  #KHz
    freq = freq[:(N//2)]

    return freq, X_mag

def stemplot(freq, X_mag):
    plt.stem(freq, X_mag, 'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.show()

if __name__ == "__main__" :
    plt.close("all")

    # sampling rate
    fs = 2048
    N = 1024

    #df = fs / N

    # sampling interval
    ts = 1.0/fs
    t = np.arange(0, N) * ts

    f1, f2, f3 = 100., 400., 700
    a1, a2, a3 = 1.0, 1.0, 1.0
    x = a1 * np.sin(2 * pi * f1 * t) + a2*np.sin(2*pi*f2*t) + a3*np.sin(2*pi*f3*t)

    plt.figure()
    plt.plot(t[:100], x[:100], 'r')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # f1, f2 = 200 / fs, 600 / fs  # normalizing freq

    freq, x_mag = mfft(x, fs)

    f1, f2 = 200, 600
    N_tap = 51
    N_tap_2 = int(N_tap / 2) + 1
    x_filt = fir2(x, f1, f2, fs, N_tap)
    # x_filt = x_filt[N_tap:]
    plt.figure()
    plt.plot(t[:100], x_filt[:100], 'r')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    freq, x_filt_mag = mfft(x_filt, fs)
    plt.figure()
    plt.subplot(211)
    stemplot(freq, x_mag)
    plt.subplot(212)
    stemplot(freq, x_filt_mag)
    plt.title("Filtered FFT")




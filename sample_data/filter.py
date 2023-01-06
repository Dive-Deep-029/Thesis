import matplotlib.pyplot as plt
import scipy
import numpy as np
import FFT_test
from scipy.fftpack import fft, ifft, fftfreq


def fft_(x,fs):
    X = fft(x)
    #print(X)
    N = len(X)  # len(X) = len(t0)
    # to recover accurate units (eg:mV)
    X_mag = np.abs(X) / N  # step1: divide fourier coefficients by N
    pos = (N // 2)
    X_mag = 2 * X_mag[:pos]  # step2: double the positive freq (negative freq we avoid)
    X_mag[0] = X_mag[0] / 2  # 0Hz and nyquist freq will not have negative side so no need to multiply by 2

  # sampling rate/freq
    ts = 1 / fs  # sampling period
    freq0 = (fftfreq(N, ts))/1000  # KHz
    #freq0 = (fftfreq(N, ts))  #Hz
    freq = freq0[:pos]

    plt.figure(figsize=(12, 6))
    plt.subplot(121)

    plt.stem(freq, X_mag, 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    #plt.xlim(0, 300)

    plt.show()

#Median filter
def median(y,t):
    median = scipy.signal.medfilt(y, kernel_size=None)
    plt.plot(t,median)
    plt.show()


#------------------------------------------------
# Create a FIR filter and apply it to x.
#------------------------------------------------
def fir1(sample_rate,x,t):
    # The Nyquist rate of the signal.
    nyq_rate = sample_rate / 2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = 5.0/nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0

    # Compute the order and Kaiser parameter for the FIR filter.
    #N, beta = scipy.signal.kaiserord(ripple_db, width)
    N= 51

    # The cutoff frequency of the filter.
    cutoff_hz = 3

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    #taps = scipy.signal.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    taps = scipy.signal.firwin(N, cutoff_hz / nyq_rate,pass_zero='lowpass')

    # Use lfilter to filter x with the FIR filter.
    filtered_x = scipy.signal.lfilter(taps, 1.0, x)

    # The phase delay of the filtered signal.
    delay = 0.5 * (N - 1) / sample_rate

    # Plot the original signal.
    plt.plot(t, x)
    # Plot the filtered signal, shifted to compensate for the phase delay.
    plt.plot(t - delay, filtered_x, 'r-')
    # Plot just the "good" part of the filtered signal.  The first N-1
    # samples are "corrupted" by the initial conditions.
    plt.plot(t[N - 1:] - delay, filtered_x[N - 1:], 'g', linewidth=4)

    plt.xlabel('t')
    plt.grid(True)

    plt.show()
    fft_(filtered_x)

def fir2(x,t,fs):

   #sampling freq
    nyqt = fs * 0.5
    f1, f2 = 200/fs, 600/fs     #normalizing freq
    N_tap = 51
    h = scipy.signal.firwin(N_tap, cutoff=[f1,f2],pass_zero=False)
    #n = np.arange(10000)
    #x_syn = np.sin(2 * np.pi * 50e3 * n / fs) + np.sin(2 * np.pi * 250e3 * n / fs)

    y_filt = scipy.signal.lfilter(h, [1.0], x)

    plt.figure()
    plt.subplot(211)
    plt.plot(t[:100], x[:100])
    plt.subplot(212)
    plt.plot(t[:100], y_filt[:100])
    plt.show()

    plt.figure()
    fft_(y_filt,fs)








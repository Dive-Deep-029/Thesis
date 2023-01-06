import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
import scipy

fs = 2e6
N_tap = 51
pi = np.pi
f1, f2 = 50e3, 600e3

def fir2(f1, f2, fs, N_tap=51):
    taps = scipy.signal.firwin(N_tap, [f1, f2], pass_zero=False, fs = fs)
    return taps

if __name__ == "__main__" :

    taps = fir2(f1, f2, fs)

    w, h = scipy.signal.freqz(taps, worN=8000)
    plt.figure()
    plt.plot((w / pi) * fs/2, np.abs(h), linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency Response')
    # plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.show()
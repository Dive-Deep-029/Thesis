import os

import matplotlib.pyplot as plt

import vallenae as vae
from scipy.fftpack import fft, ifft, fftfreq

import numpy as np


#HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
HERE = "D:\\Thesis\\acoustics\\Data collected\\25-10-2022\\sensor directly on lens"
TRADB = os.path.join(HERE, "scratch test.tradb")  # uncompressed
TRAI = 4  # just an example, no magic here


def main():
    # Read waveform from tradb
    with vae.io.TraDatabase(TRADB) as tradb:
        y, t = tradb.read_wave(TRAI)

    rmss = vae.features.rms(y)
    print("rms", rmss)

    y1 = y*1e3  # in mV
    t1 = t*1e6  # for µs

    X = fft(y1)
    N = len(X)  # No. of samples
    delta_t = t[2]-t[1]
    #print("N, delta_t", N, delta_t)
    freq = fftfreq(N, delta_t)

    rms = []
    for i in range(N):
        rms.append(abs(X[i]*0.707))
    #print("rms", rms)
    plt.plot(t, rms, marker='o')
    plt.xlim(0, 0.001)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(121)

    # x[start:end:step] default values  x[0:len(x):1]
   # plt.stem(freq[:N//2], np.abs(X[:N//2]), 'b', \
     #        markerfmt=" ", basefmt="-b")
    plt.plot(freq,abs(X))
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 250000)

    plt.subplot(122)
    plt.plot(t, ifft(X), 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
import os

import matplotlib.pyplot as plt

import vallenae as vae

# from scipy.fftpack import fft, ifft

# import numpy as np


# HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
HERE = "D:\Thesis\Acoustics\Python"
TRADB = os.path.join(HERE, "sample_plain.tradb")  # uncompressed
TRAI = 4  # just an example, no magic here -  trai: Transient recorder index (unique key between pridb and tradb)


def main():
    # Read waveform from tradb
    with vae.io.TraDatabase(TRADB) as tradb:
        y, t = tradb.read_wave(TRAI, time_axis=False)
        # a,b = tradb.read_continuous_wave(1)

    y *= 1e3  # in mV
    # t *= 1e6  # for µs
    print('time', t)
    #print('first and last value of time', t[0], t[-1])


    # Plot waveforms
    plt.figure(figsize=(8, 4), tight_layout=True)
    plt.plot(t, y)
    plt.xlabel("Time [µs]")
    plt.ylabel("Amplitude [mV]")
    plt.title(f"Transient Wave Plot; trai = {TRAI}")
    plt.show()


if __name__ == "__main__":
    main()
"""
Read and plot transient data
============================
"""

import os

import matplotlib.pyplot as plt

import vallenae as vae

import filter

from scipy import signal

import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, rfft

#HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
HERE = "D:\\Thesis\\acoustics\\Data collected\\25-10-2022\\sensor directly on lens"
TRADB = os.path.join(HERE, "scratch test.tradb")  # uncompressed
TRAI = 13  # just an example, no magic here


def getdata():
    # Read waveform from tradb
    with vae.io.TraDatabase(TRADB) as tradb:
        y, t = tradb.read_wave(TRAI)

    y *= 1e3  # in mV
    t *= 1e6  # for µs

    # Plot waveforms
    plt.figure(figsize=(8, 4), tight_layout=True)
    plt.plot(t, y)
    plt.xlabel("Time [µs]")
    plt.ylabel("Amplitude [mV]")
    plt.title(f"Transient Wave Plot; trai = {TRAI}")
    plt.show()
    return y, t


if __name__ == "__main__":
    x, t = getdata()
    filter.fir2(t,x)


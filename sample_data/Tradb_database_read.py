"""
Read and plot transient data
============================
"""

import os

import matplotlib.pyplot as plt

import vallenae as vae
from scipy.fftpack import fft, ifft

import numpy as np

# HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
HERE = "D:\Thesis\Python"
TRADB = os.path.join(HERE, "sample.tradb")  # uncompressed
TRAI = 4  # just an example, no magic here


def main():
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


if __name__ == "__main__":
    main()


"""
Read and plot transient data
============================
"""

import os
import matplotlib.pyplot as plt
import vallenae as vae

# Will take the path of the database from local path
HERE = "D:\Thesis\\acoustics\Data collected\9-11-2022\Wo_continous_mode_with_lens"
TRADB = os.path.join(HERE, "one_scratch4.tradb")
TRAI = 1   #Transient record index  or window number = 10000 µs

def main():
    # Read waveform from tradb
    with vae.io.TraDatabase(TRADB) as tradb:
        y0, t0 = tradb.read_wave(TRAI)

    x = y0 * 1e3  # in mV
    t = t0 * 1e6  # for µs

    # Plot waveforms
    plt.plot(t, x)
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"Transient Wave Plot; trai = {TRAI}")
    plt.show()


if __name__ == "__main__":
    main()

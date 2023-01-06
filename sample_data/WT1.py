from pywt import wavedec


import os

import matplotlib.pyplot as plt

import vallenae as vae
import scipy
import numpy as np
import pywt

pi = np.pi


#HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
HERE = "D:\Thesis\\acoustics\Data collected\\23-11-2022\water"
TRADB = os.path.join(HERE, "sensor_5cm_from_lens_CM.tradb")  # uncompressed
TRAI = 850
fs = 2000000


# Read waveform from tradb

with vae.io.TraDatabase(TRADB) as tradb:
    y0, t0 = tradb.read_wave(TRAI)

x = y0 * 1e3  # in mV
t = t0 * 1e6  # for µs

wave_type = "db2"
coeffs = pywt.wavedec(x, wave_type, level=2)
cA2, cD2, cD1 = coeffs

plt.close("all")
plt.figure()
plt.subplot(411)
plt.plot(x)
plt.subplot(412)
plt.plot(cA2)
plt.subplot(413)
plt.plot(cD2)
plt.subplot(414)
plt.plot(cD1)
plt.show()


print(len(cA2), len(cD2), len(cD1))

czD2 = cD2 * 0
czD1 = cD1 * 0

new_coeff = [cA2, czD2, czD1]
xw = pywt.waverec(new_coeff, wave_type)

plt.figure()

plt.subplot(211)
plt.plot(x)
plt.subplot(212)
plt.plot(xw)
plt.show()
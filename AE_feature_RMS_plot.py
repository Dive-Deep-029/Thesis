"""
AE features plotting - RMS and Maximum amplitude in FFT

Filters - Bandpass filter, Median filter, Wavelet filter
============================
"""


import os
import matplotlib.pyplot as plt
import vallenae as vae
import numpy as np
from scipy.fftpack import fft,fftfreq
import scipy
import pywt


# Will take the path of the database from local path
HERE = "D:\Thesis\\acoustics\Data collected\\08-02-23"
TRADB = os.path.join(HERE, "NBK7_700_0_7_30min.tradb")
PRIDB = os.path.join(HERE, "NBK7_700_0_7_30min.pridb")
pridb = vae.io.PriDatabase(PRIDB)
df_hits = pridb.read_hits()
TRAI_list = df_hits.trai.tolist()


TRAI_list = TRAI_list[:259]
TRAI_list = [x for x in TRAI_list if x % 2 != 0]  # Channel 1
#TRAI_list = [x for x in TRAI_list if x%2==0]     # Channel 2
print(TRAI_list)
print(len(TRAI_list))


x_max_list_avg = []
rms = []
rms_averages = []
x_full = []
x_max_list = []
freq_of_max_ampl = []
freq_range = []

fs = 2e6
N_tap = 121
pi = np.pi
f1, f2 = 35e3, 40e3


def WT(x1):

    wave_type = "db16"
    level = 3
    coeffs = pywt.wavedec(x1, wave_type, level=level)

    cA3, cD3, cD2, cD1 = coeffs

    # Calculate standard deviations of each coefficient
    cA3_std = np.std(cA3)
    cD3_std = np.std(cD3)
    cD2_std = np.std(cD2)
    cD1_std = np.std(cD1)

    # Choose a threshold factor
    threshold_factor = 0.1

    # Calculate threshold values for each coefficient
    cA3_threshold = cA3_std * threshold_factor
    cD3_threshold = cD3_std * threshold_factor
    cD2_threshold = cD2_std * threshold_factor
    cD1_threshold = cD1_std * threshold_factor

    # Set values below the threshold to zero
    czA3 = cA3 * (cA3 > cA3_threshold)
    czD3 = cD3 * (cD3 > cD3_threshold)
    czD2 = cD2 * (cD2 > cD2_threshold)
    czD1 = cD1 * (cD1 > cD1_threshold)

    # Reconstruct the filtered signal using the modified coefficients
    new_coeff = [czA3,czD3, czD2, czD1]
    x2 = pywt.waverec(new_coeff, wave_type)
    return x2

def fir2(f1, f2, fs, N_tap):
    taps = scipy.signal.firwin(N_tap, [f1, f2], pass_zero=False, fs = fs)
    return taps

def median(x3):
    x4 = scipy.signal.medfilt(x3, kernel_size=None)
    return x4

def HFFT(x4):
    X = fft(x4)
    N = len(X)  # len(X) = len(t0)
    # to recover accurate units (eg:mV)
    X_mag = np.abs(X) / N  # step1: divide fourier coefficients by N
    pos = (N // 2)
    X_mag = 2 * X_mag[:pos]  # step2: double the positive freq (negative freq we avoid)
    X_mag[0] = X_mag[0] / 2  # 0Hz and nyquist freq will not have negative side so no need to multiply by 2

    fs = 2e6  # sampling rate/freq
    ts = 1 / fs  # sampling period
    freq0 = (fftfreq(N, ts)) / 1000  # KHz
    freq = freq0[:pos]

    X_FFT_max = np.max(X_mag)


    # max amplitude and coreesponding freq
    X_mag_sorted = X_mag
        # sort the array in descending order
    X_mag_sorted = sorted(X_mag_sorted, reverse=True)
        # return the element at the first position of the sorted array
    X_max = X_mag_sorted[0]
    index = np.where(X_mag == X_max)
    freq_of_max_ampl.append((round((float(freq[index])), 2)))

    return X_FFT_max,freq_of_max_ampl


for TRAI in TRAI_list:
    with vae.io.TraDatabase(TRADB) as tradb:
        y0, t0 = tradb.read_wave(TRAI)

    x1 = y0 * 1e3  # in mV
    t = t0 * 1e6  # for µs

    x2 = WT(x1)
    taps = fir2(f1, f2, fs, N_tap)
    x3 = scipy.signal.lfilter(taps, 1, x2)
    x4 = median(x3)

    x_full.append(x4)

    rms.append(np.sqrt(np.mean(np.square(x4))))

    X_FFT_max,freq_of_max_ampl = HFFT(x4)
    x_max_list.append(X_FFT_max)
    #freq_range.append(freq_of_max_ampl)


overall_rms = np.mean(rms)
avg_freq_range = np.mean(freq_of_max_ampl)


chunk_size = 50
# create an empty list to store the averages
# split the signal into chunks
for i in range(0, len(rms), chunk_size):
    chunk = rms[i:i + chunk_size]
    # calculate the average of the chunk and append it to the list
    rms_averages.append(np.mean(chunk))

plt.subplot(211)
plt.title(f'Averaged RMS-50 points_Sensor1, Mean_RMS= {overall_rms}, Frequency_range= {avg_freq_range}')
plt.xlabel("Window index")
plt.ylabel("Amplitude RMS [mV]")
plt.grid()
plt.plot(rms_averages)


chunk_size = 50
# create an empty list to store the averages
# split the signal into chunks
for i in range(0, len(x_max_list), chunk_size):
    chunk = x_max_list[i:i + chunk_size]
    # calculate the average of the chunk and append it to the list
    x_max_list_avg.append(np.mean(chunk))

print("freq_of_max_ampl", freq_of_max_ampl)
print(len(x_max_list))
plt.subplot(212)
plt.title("Averaged FFT maximum amplitudes-50 points_Sensor1")
plt.xlabel("Window index")
plt.ylabel("Averaged_x_max_FFT")
plt.grid()
plt.plot(x_max_list_avg)
plt.tight_layout()
plt.show()

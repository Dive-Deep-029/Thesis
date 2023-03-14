# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 12:00:00 2023

@author: Michael
"""
#necessary modules - clean up here
import tkinter as tk
from tkinter import filedialog
import csv
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from mpl_toolkits import mplot3d 
from scipy import stats
from scipy.fftpack import fft, fftfreq
from scipy.fftpack import rfft, rfftfreq

#select file with dialog
file_path = filedialog.askopenfilename()
datei=file_path

#set up values array
values = []
with open(datei, "r") as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=';')
    next(csv_reader)
    header = next(csv_reader)
    for i in csv_reader:
        values.append(i)


power = []

for i in range(0,len(values)):
    #num.append(float(values[vergleich][0]))
    x = values[i][2]
    x = x.replace(',','.')
    power.append(float(x))

#time in seconds - four values are measured per second
time_process = np.linspace(1,len(values)+1,len(power))*0.25

#Plot the raw measured date
plt.scatter(time_process,power)
plt.xlabel('Process time [s]')
plt.ylabel('Power [W]')
plt.grid()
plt.show()

#Calculate the derivation of the signal = power change rate
power_derivation = np.diff(power)
time_process_derivation = np.linspace(1,len(power_derivation)+1,len(power)-1)*0.25

plt.scatter(time_process_derivation,power_derivation)
plt.xlabel('Process time [s]')
plt.ylabel('Power change rate [W/s]')
plt.grid()
plt.show()

#This criterion delivers, which change of the slope is relevant for starting the evaluation
termination_criterion = 2
relevant_slopes = np.where(abs(power_derivation)>termination_criterion)
begin_signal = relevant_slopes[0][0]
end_signal = relevant_slopes[0][len(relevant_slopes[0])-1]
baseline = np.mean(power[0:begin_signal-10])

#Subtract the baseline and calculate the process complete power, mean power and standard deviation
power_without_baseline = np.subtract(power,baseline)
complete_process_power = np.trapz(power_without_baseline[begin_signal:end_signal])
mean_power = np.mean(power_without_baseline[begin_signal:end_signal])
standard_deviation_power = np.std(power_without_baseline[begin_signal:end_signal])

#Linear regression over complete signal length
slope, intercept, r, p, std_err = stats.linregress(time_process[begin_signal:end_signal], power_without_baseline[begin_signal:end_signal])

power_linear = []

#Calculate the straight line - based on linear regression data
for i in range(0,len(time_process)):
    power_linear_single = slope * time_process[i] + intercept
    power_linear.append(power_linear_single)

#Plot the picture
plt.plot(time_process[begin_signal:end_signal],power[begin_signal:end_signal])
plt.plot(time_process[begin_signal:end_signal],power_without_baseline[begin_signal:end_signal])
plt.plot(time_process[begin_signal:end_signal],power_linear[begin_signal:end_signal], c='g')
plt.title('Polishing experiment - Process Power')
plt.xlabel('Time [s]')
plt.ylabel('Power [W]')
plt.legend(['Power', 'Power without baseline', 'Power (linear regression)'])
plt.grid()
plt.show()

#Generate FFT data
N = len(power_without_baseline[begin_signal:end_signal])
sampling_rate = 4
power_fft = rfft(power_without_baseline[begin_signal:end_signal])
x_fft = rfftfreq(N, 1 / sampling_rate)

#plot fft date
plt.plot(x_fft,np.abs(power_fft))
plt.yscale('log')
plt.title('FFT Power Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('log. FFT Power')
plt.grid()
plt.show()

#Subtract real values from regression calculated values
power_difference_mean = np.subtract(power_without_baseline, mean_power)

begin_signal = np.argmax(power_difference_mean)

plt.plot(time_process[begin_signal:end_signal], power_difference_mean[begin_signal:end_signal])
plt.grid()
plt.plot()

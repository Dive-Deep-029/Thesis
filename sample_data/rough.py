import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fftpack import fft, ifft, fftfreq

def fir2(f1, f2, fs, N_tap=51):
    taps = scipy.signal.firwin(N_tap, [f1, f2], pass_zero=False, fs = fs)
    # Return taps as tuple with gain of 1
    return (taps, 1)

fs = 2e6
N_tap = 51
pi = np.pi
f1, f2 = 200e3, 600e3

# Get taps and gain
taps, gain = fir2(f1, f2, fs)

# Create LTI system object
system = scipy.signal.lti(taps, gain)

# Calculate impulse response
t, y = scipy.signal.impulse(system)

# Plot impulse response
plt.plot(t, y)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Impulse Response')
plt.grid(True)
plt.show()


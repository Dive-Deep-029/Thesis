# plt.figure(figsize=(8, 4), tight_layout=True)
    # plt.plot(b, a)
    # plt.xlabel("Time [µs]")
    # plt.ylabel("Amplitude [mV]")
    # plt.title(f"Transient Wave Plot; trai = {TRAI}")
    # plt.show()
    print('y',y)
    X = fft(y)
    print("Complex valued sample fft",X)
    N = len(t)
    print('length of fft array', N)
    n = np.arange(N)
    print("array from 0 to N", n)
    t = n/2000
    #N = 512
    fs = n/t

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(fs, X)
    # plt.stem(fs, np.abs(X), 'b', \
    # markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    # plt.xlim(0, 10)

    plt.subplot(122)
    plt.plot(t, ifft(X), 'r')
    plt.xlabel('Time (s)')    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()




# Plot waveforms
    plt.figure(figsize=(8, 4), tight_layout=True)
    plt.plot(t, y)
    plt.xlabel("Time [µs]")
    plt.ylabel("Amplitude [mV]")
    plt.title(f"Transient Wave Plot; trai = {TRAI}")
    plt.show()
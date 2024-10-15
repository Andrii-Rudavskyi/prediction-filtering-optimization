import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def extract_noise(signal, axt, dt, pl=0, dataLabel = ''):
    # Step 1: Create the filter manually
    alpha = 1 - np.exp(-np.diff(axt) / dt)  # Compute alpha for each interval
    alpha = np.insert(alpha, 0, alpha[0])  # Match the length with the signal

    # Initialize filtered signal
    filtered_signal = np.zeros_like(signal)
    filtered_signal[0] = signal[0]  # First value remains the same

    # Step 2: Apply the filter using a recursive approach
    for i in range(1, len(signal)):
        filtered_signal[i] = alpha[i] * signal[i] + (1 - alpha[i]) * filtered_signal[i - 1]

    # Step 3: Calculate the noise
    noise = signal - filtered_signal

    if pl:
        # Plot the results
        plt.figure(figsize=(10, 8))

        plt.subplot(3, 1, 1)
        plt.plot(axt, signal)
        plt.title('Original Signal, ' + dataLabel )
        plt.ylim([np.min(signal), np.max(signal)])

        plt.subplot(3, 1, 2)
        plt.plot(axt, filtered_signal)
        plt.title('Filtered Signal, ' + dataLabel)
        plt.ylim([np.min(signal), np.max(signal)])

        plt.subplot(3, 1, 3)
        plt.plot(axt, noise)
        plt.title(f'Noise -- std = {np.std(noise):.4f}')

        plt.tight_layout()
        plt.show()

    return filtered_signal, noise

def extract_noise_2(signal, time, windowSize = 5, polynomialOrder = 2):
    #Smooths signal by appling moving window. Points withing the window are fitted with the polynomial and the middle
    # of the window is estimated by calculating polynom value at that point.
    filtered_signal = []
    filtered_time = []
    for i in range(0, len(signal) - windowSize + 1):
        t = time[i:i + windowSize]
        x = signal[i:i + windowSize]
        p = np.polyfit(t, x, polynomialOrder)

        t_est = t[i + (windowSize - 1)/2]
        x_est = 0
        for j in range(0, polynomialOrder + 1):
            x_est = x_est + p[j] * np.power(t_est, polynomialOrder - j)

        filtered_time = np.append(filtered_time, t_est)
        filtered_signal = np.append(filtered_signal, x_est)

    filtered_noise = signal[int((windowSize - 1) / 2):int(len(signal) - (windowSize - 1) / 2)] - filtered_signal
    return filtered_time, filtered_signal, filtered_noise


dataFolder = './data/'

for filename in os.listdir(dataFolder):
    if filename.endswith('cnsdk_blink_raw_data_trace.csv'):
        eyetrackerdata = pd.read_csv(dataFolder + filename)
        imudata = pd.read_csv(dataFolder + filename)

        eyetrackerdata_timestapms = eyetrackerdata[' cameraTimestamp']
        eyetrackerdata_timestapms = eyetrackerdata_timestapms - eyetrackerdata_timestapms[0]
        leftEyeX = eyetrackerdata[' leftEyeX']
        leftEyeY = eyetrackerdata[' leftEyeY']
        leftEye3DZ = eyetrackerdata[' leftEye3DZ']

        # filtered_signal, noise = extract_noise(
        #     leftEyeX, eyetrackerdata_timestapms, dt=10.0
        # )

        filtered_time, filtered_signal, noise = extract_noise_2(leftEyeX, eyetrackerdata_timestapms)

        plt.figure(figsize=(10, 8))
        plt.subplot(3, 1, 1)
        plt.plot(eyetrackerdata_timestapms, leftEyeX, '-o')
        plt.title('Signal')

        plt.subplot(3, 1, 2)
        plt.plot(filtered_time, filtered_signal)
        plt.title('Filtered signal')

        plt.subplot(3, 1, 3)
        plt.plot(filtered_time, noise)
        plt.title('Noise')

        plt.show()


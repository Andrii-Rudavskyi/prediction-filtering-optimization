import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sn

from helper2 import calculate_delay_error, calculate_predicted_error, predict

dataFolder = './data/'
imagesFolder = './images/'

latency = 0.1

# Focal distance to convert to angular space. TODO: Load it properly from intrinsics (product/device specific).
fx = 330
fy = 330

for filename in os.listdir(dataFolder):
    if filename.endswith('cnsdk_blink_raw_data_trace.csv'):
        eyetrackerdata = pd.read_csv(dataFolder + filename)
        imudata = pd.read_csv(dataFolder + filename)

        eyetrackerdata_timestapms = eyetrackerdata[' cameraTimestamp']
        #Start trace from zero
        eyetrackerdata_timestapms = 0.001 * (eyetrackerdata_timestapms - eyetrackerdata_timestapms[0])

        #Convert to angular space
        leftEyeX = eyetrackerdata[' leftEyeX'] / fx
        leftEyeY = eyetrackerdata[' leftEyeY'] / fy
        leftEye3DZ = eyetrackerdata[' leftEye3DZ']

        signal = [leftEyeX, leftEyeY, leftEye3DZ]
        signal_label = ['leftEyeX', 'leftEyeY', 'leftEye3DZ']

        #filtered_time, filtered_signal, noise = extract_noise_2(signal[0], eyetrackerdata_timestapms)
        delay_error = calculate_delay_error(eyetrackerdata_timestapms, signal[0], predictionTime=latency)
        predicted = predict(eyetrackerdata_timestapms, signal[0], predcictionTime=latency)
        ground_truth, predicted_error = calculate_predicted_error(eyetrackerdata_timestapms, signal[0], predicted, predictionTime=latency)
        fig, ax1 = plt.subplots()

        ax1.plot(eyetrackerdata_timestapms, signal[0], color='blue')
        ax1.plot(eyetrackerdata_timestapms, ground_truth, color='black')
        ax1.plot(eyetrackerdata_timestapms, predicted, color='g')
        ax1.legend(['Original delayed signal', 'Ground truth signal', 'Fully latency compensated signal'])
        ax1.set_ylabel("Signal, signal units")

        ax2 = ax1.twinx()
        ax2.plot(eyetrackerdata_timestapms, delay_error, '--', color='r', linewidth=1.0)
        ax2.plot(eyetrackerdata_timestapms, predicted_error, marker='.', color="red", mfc='none', linewidth=0.5)
        ax2.legend(['Delay error (without prediction)', 'Error after full latency compensation (last point velosity model)'])
        ax2.set_ylabel("Error, signal units", color='r')

        plt.show()



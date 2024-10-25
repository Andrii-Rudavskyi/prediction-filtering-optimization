import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sn

from helper2 import calculate_delay_error, calculate_predicted_error, predict, PredictionModel, extract_noise_2, PredictionModelType

dataFolder = './data/'
imagesFolder = './images/'

# Set latency in seconds
latency = 0.10

# Focal distance to convert to angular space. TODO: Load it properly from intrinsics (product/device specific).
fx = 330
fy = 330

#selection of the prediction model
predictionModel = PredictionModel(PredictionModelType.PolynomialFit)
if (predictionModel.modelType == PredictionModelType.PolynomialFit):
    predictionModel.setBufferSize(6)
    predictionModel.setPolynomialOrder(2)

for filename in os.listdir(dataFolder):
    if filename.endswith('cnsdk_blink_raw_data_trace.csv'):
        #Extract trace from file
        eyetrackerdata = pd.read_csv(dataFolder + filename)

        eyetrackerdata_timestapms = eyetrackerdata[' cameraTimestamp']
        #Start trace from zero
        eyetrackerdata_timestapms = 0.001 * (eyetrackerdata_timestapms - eyetrackerdata_timestapms[0])

        #Convert to angular space
        leftEyeX = eyetrackerdata[' leftEyeX'] / fx
        leftEyeY = eyetrackerdata[' leftEyeY'] / fy
        leftEye3DZ = eyetrackerdata[' leftEye3DZ']

        signal = [leftEyeX, leftEyeY, leftEye3DZ]
        signal_label = ['leftEyeX', 'leftEyeY', 'leftEye3DZ']

        for i in range(0, 3):
            # filtered_time, filtered_signal, noise = extract_noise_2(signal[0], eyetrackerdata_timestapms)
            delay_error = calculate_delay_error(eyetrackerdata_timestapms, signal[i], predictionTime=latency)

            predicted = predict(eyetrackerdata_timestapms, signal[i], predictionModel, predcictionTime=1.0 * latency)
            # predicted = predict(eyetrackerdata_timestapms, signal[0], PredictionModel.PolynomialFit, predcictionTime=latency)
            ground_truth, predicted_error = calculate_predicted_error(eyetrackerdata_timestapms, signal[i], predicted,
                                                                      predictionTime=latency)

            plt.subplot(3, 2, 2 * i + 1)
            plt.plot(eyetrackerdata_timestapms, signal[i], color='blue')
            plt.plot(eyetrackerdata_timestapms, ground_truth, color='black')
            plt.plot(eyetrackerdata_timestapms, predicted, color='g')
            plt.title('Signal:  ' + signal_label[i])
            plt.legend(['Original delayed signal', 'Ground truth signal', 'Fully latency compensated signal'])
            plt.ylabel("Signal, signal units")

            plt.subplot(3, 2, 2 * i + 2)
            plt.plot(eyetrackerdata_timestapms, delay_error, '--', color='b', linewidth=2.0)
            plt.plot(eyetrackerdata_timestapms, predicted_error, marker='.', color="red", mfc='none', linewidth=0.5)
            plt.title('Error:  ' + signal_label[i])
            plt.legend(['Delay error (without prediction)', 'Error after full latency compensation (last point velosity model)'])
            plt.ylabel("Error, signal units")

        plt.show()



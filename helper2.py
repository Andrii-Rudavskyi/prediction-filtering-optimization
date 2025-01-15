import numpy as np
from enum import Enum

VD = 600
slant = 0.33
IPD = 63

ph0 = 0

class PredictionModelType(Enum):
    LastVelocity = 0,
    PolynomialFit = 1

class PredictionModel:
    def __init__(self, modelType = PredictionModelType.PolynomialFit):
        self.modelType = modelType
        self.polynomialOrder = 2
        self.n_buffers = 3
        print("Prediction model", self.modelType)

    def setPolynomialOrder(self, polynomialOrder):
        if self.modelType == PredictionModelType.PolynomialFit:
            self.polynomialOrder = polynomialOrder
        else:
            raise Exception("The chosen model does not support this parameter. This parameter will not be used")

    def setBufferSize(self, n_buffers):
        if self.modelType == PredictionModelType.PolynomialFit:
            self.n_buffers = n_buffers
        else:
            raise Exception("The chosen model does not support this parameter. This parameter will not be used")




def calculate_phase_error(xc, yc, deltaZ, deltaX2d, deltaY2d, Z, x0, y0):
    donopx = VD / (2 * IPD)
    phase_error = -donopx * (((xc - x0) + slant * (yc - y0)) * deltaZ / np.power(Z, 2) + deltaX2d + slant * deltaY2d)
    return phase_error

def extract_noise_2(signal, time, windowSize = 5, polynomialOrder = 2):
    #Smooths signal by appling moving window. Points withing the window are fitted with the polynomial and the middle
    # of the window is estimated by calculating polynom value at that point.
    filtered_signal = []
    filtered_time = []

    origin = time[0]
    time = time - origin # Shift to origin for better numerical conditioning
    for i in range(0, len(signal) - windowSize + 1):
        t = time[i:i + windowSize]
        x = signal[i:i + windowSize]
        p = np.polyfit(t, x, polynomialOrder)

        t_est = t[int((windowSize - 1)/2)]
        x_est = 0
        for j in range(0, polynomialOrder + 1):
            x_est = x_est + p[j] * np.power(t_est, polynomialOrder - j)

        filtered_time = np.append(filtered_time, t_est)
        filtered_signal = np.append(filtered_signal, x_est)

    filtered_noise = signal[int((windowSize - 1) / 2):int(len(signal) - (windowSize - 1) / 2)] - filtered_signal

    filtered_time = filtered_time + origin
    return filtered_time, filtered_signal, filtered_noise

def calculate_delay_error(raw_t, raw_signal, predictionTime = 0.1):
    # just shift in time by expected latency. Once prediction filters applied another function should be implemented that calculates residual error and probably another that calculates noise.
    predicted_t = raw_t - predictionTime
    error = raw_signal - np.interp(raw_t, predicted_t, raw_signal)
    return error

def calculate_predicted_error(raw_t, raw_signal, predicted_signal, predictionTime = 0.1):
    # just shift in time by expected latency. Once prediction filters applied another function should be implemented that calculates residual error and probably another that calculates noise.
    #raw_t and raw signal are the raw data avaliable after certain delay
    # predicted_signal is the
    predicted_t = raw_t - predictionTime
    ground_truth = np.interp(raw_t, predicted_t, raw_signal)
    error = predicted_signal - ground_truth
    return ground_truth, error

def predict(time, signal, predictionModel, predcictionTime = 0.1):
    n_buffer = predictionModel.n_buffers
    polynomialOrder = predictionModel.polynomialOrder
    predicted = []

    if (predictionModel.modelType == PredictionModelType.LastVelocity):
        for i in range(0, len(time)):
            if i < 1:
                predicted.append(signal[i])
            else:
                t1 = time[i-1]
                t2 = time[i]
                x1 = signal[i - 1]
                x2 = signal[i]
                v = (x2 - x1) / (t2 - t1)
                x_est = x2 + v * predcictionTime
                predicted.append(x_est)
    elif (predictionModel.modelType == PredictionModelType.PolynomialFit):
        for i in range(0, len(time)):
            if i < n_buffer:
                predicted.append(signal[i])
            else:
                t = time[i - n_buffer:i]
                x = signal[i - n_buffer:i]
                t_est = t[i - 1] + predcictionTime
                p = np.polyfit(t, x, polynomialOrder)
                x_est = 0
                for j in range(0, polynomialOrder + 1):
                    x_est = x_est + p[j] * np.power(t_est, polynomialOrder - j)
                predicted.append(x_est)
    else:
        print('Model not supported')



    return predicted
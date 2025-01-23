from LLSfilter import LLSfilter, LLSFilterParameters, FilterType, PolynomialFilterParameters, calculate_prediction_error
import matplotlib.pyplot as plt
import numpy as np
from helper2 import extract_noise_2
import os

data_labels = ['0 cm/s, 0 lux', '150 cm/s, 0 lux', '250 cm/s, 0 lux', '350 cm/s, 0 lux',
          '0 cm/s, 40 lux', '150 cm/s, 40 lux', '250 cm/s, 40 lux', '350 cm/s, 40 lux',
          '0 cm/s, 90 lux', '150 cm/s, 90 lux', '250 cm/s, 90 lux', '350 cm/s, 90 lux']

dataPath = './data/windows_traces/noise/'

t_continer = []
x_container = []
filtered_time_x_container = []
filtered_signal_x_container = []
filtered_noise_x_container = []
noise = []

predicted_time_container = []
predicted_container = []
filtered_predicted_noise_x_container = []
filtered_predicted_time_x_container = []
filtered_predicted_signal_x_container = []
predicted_noise = []

n_avergaing = 200
n_predicted_avergaing = 800

for dirname in os.listdir(dataPath):
    llsFilterParameters = LLSFilterParameters(dataPath=dataPath + dirname + '/', useNoiseRejection=True, speedLimit_cm_s=[1000000, 1000000, 1000000])
    llsfilter = LLSfilter(llsFilterParameters, debuPlots=False)

    t, x, y, z = llsfilter.retrieveRawData(dataPath=dataPath + dirname + '/')
    time_origin = t[0]
    t = [ti - time_origin for ti in t]

    x = [xi * 10 for xi in x] #convert to mm

    filtered_time_x, filtered_signal_x, filtered_noise_x = extract_noise_2(x, t, windowSize=21, polynomialOrder=5)
    # filtered_time_y, filtered_signal_y, filtered_noise_y = extract_noise_2(y, t)
    # filtered_time_z, filtered_signal_z, filtered_noise_z = extract_noise_2(z, t)

    t_predicted, x_predicted, y_predicted, z_predicted = llsfilter.outputThread(dataPath=dataPath + dirname + '/')
    t_predicted = t_predicted - time_origin
    x_predicted = [xi * 10 for xi in x_predicted] #convert to mm
    filtered_predicted_time_x, filtered_predicted_signal_x, filtered_predicted_noise_x = extract_noise_2(x_predicted, t_predicted, windowSize=81, polynomialOrder=5)

    predicted_time_container.append(t_predicted)
    predicted_container.append(x_predicted)

    average_noise = []
    average_predicted_noise = []

    for i in range(n_avergaing, len(filtered_signal_x)):
        #average_noise.append(np.average(np.abs(filtered_noise_x[i - n_avergaing:i])))
        average_noise.append(np.std(filtered_noise_x[i - n_avergaing:i]))

    noise.append(average_noise)

    for i in range(n_predicted_avergaing, len(filtered_predicted_time_x)):
        average_predicted_noise.append(np.std(filtered_predicted_noise_x[i - n_predicted_avergaing:i]))

    predicted_noise.append(average_predicted_noise)

    t_continer.append(t)
    x_container.append(x)
    filtered_time_x_container.append(filtered_time_x)
    filtered_signal_x_container.append(filtered_signal_x)
    filtered_noise_x_container.append(filtered_noise_x)

    filtered_predicted_time_x_container.append(filtered_predicted_time_x)
    filtered_predicted_signal_x_container.append(filtered_predicted_signal_x)
    filtered_predicted_noise_x_container.append(filtered_predicted_noise_x)


fig, axs = plt.subplots(3, 4)
for i in range(0, len(t_continer)):

    axs[i // 4, i % 4].plot(filtered_time_x_container[i], filtered_signal_x_container[i])
    axs[i // 4, i % 4].plot(t_continer[i], x_container[i], '-o')
    axs[i // 4, i % 4].plot(predicted_time_container[i], predicted_container[i])
    axs[i // 4, i % 4].plot(filtered_predicted_time_x_container[i], filtered_predicted_signal_x_container[i])
    axs[i // 4, i % 4].set_title(data_labels[i])

    ax2 = axs[i // 4, i % 4].twinx()
    #ax2.plot(filtered_time_x_container[i], filtered_noise_x_container[i])
    ax2.plot(filtered_predicted_time_x_container[i], filtered_predicted_noise_x_container[i])

plt.show()


fig, axs = plt.subplots(3, 4)
for i in range(0, len(noise)):
    axs[i // 4, i % 4].plot(noise[i])
    axs[i // 4, i % 4].plot(predicted_noise[i])
    axs[i // 4, i % 4].set_title(data_labels[i])
plt.show()

fig, axs = plt.subplots(3, 4)
for i in range(0, len(noise)):
    #axs[i // 4, i % 4].hist(filtered_noise_x_container[i], bins=20, color='skyblue', edgecolor='black')
    axs[i // 4, i % 4].hist(filtered_predicted_noise_x_container[i], bins=20, color='skyblue', edgecolor='black')
    axs[i // 4, i % 4].set_title(data_labels[i])
plt.show()

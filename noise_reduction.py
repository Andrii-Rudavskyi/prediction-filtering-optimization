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

n_avergaing = 200

for dirname in os.listdir(dataPath):
    llsFilterParameters = LLSFilterParameters(dataPath=dataPath + dirname + '/')
    llsfilter = LLSfilter(llsFilterParameters, debuPlots=False)

    t, x, y, z = llsfilter.retrieveRawData(dataPath=dataPath + dirname + '/')
    time_origin = t[0]
    t = [ti - time_origin for ti in t]

    x = [xi * 10 for xi in x] #convert to mm

    filtered_time_x, filtered_signal_x, filtered_noise_x = extract_noise_2(x, t)
    # filtered_time_y, filtered_signal_y, filtered_noise_y = extract_noise_2(y, t)
    # filtered_time_z, filtered_signal_z, filtered_noise_z = extract_noise_2(z, t)

    average_noise = []

    for i in range(n_avergaing, len(filtered_signal_x)):
        average_noise.append(np.average(np.abs(filtered_noise_x[i - n_avergaing:i])))

    noise.append(average_noise)

    t_continer.append(t)
    x_container.append(x)
    filtered_time_x_container.append(filtered_time_x)
    filtered_signal_x_container.append(filtered_signal_x)
    filtered_noise_x_container.append(filtered_noise_x)



fig, axs = plt.subplots(3, 4)
for i in range(0, len(t_continer)):

    axs[i // 4, i % 4].plot(filtered_time_x_container[i], filtered_signal_x_container[i])
    axs[i // 4, i % 4].plot(t_continer[i], x_container[i], '-o')
    axs[i // 4, i % 4].set_title(data_labels[i])

    ax2 = axs[i // 4, i % 4].twinx()
    ax2.plot(filtered_time_x_container[i], filtered_noise_x_container[i])

plt.show()


fig, axs = plt.subplots(3, 4)
for i in range(0, len(noise)):
    axs[i // 4, i % 4].plot(noise[i])
    axs[i // 4, i % 4].set_title(data_labels[i])
plt.show()

fig, axs = plt.subplots(3, 4)
for i in range(0, len(noise)):
    axs[i // 4, i % 4].hist(noise[i], bins=10, color='skyblue', edgecolor='black')
    axs[i // 4, i % 4].set_title(data_labels[i])
plt.show()

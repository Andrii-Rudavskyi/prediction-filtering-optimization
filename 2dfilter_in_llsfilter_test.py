from LLSfilter import LLSfilter, LLSFilterParameters, FilterType, PolynomialFilterParameters, calculate_prediction_error
from StereoEyePositionFilter import StereoFilterParameters
import matplotlib.pyplot as plt
import numpy as np
from helper2 import extract_noise_2

dataPath = './data/windows_traces/noise/'
dataID = '2025-01-13___15_34_15'
dataPath = dataPath + dataID + '/'

n_avergaing = 200
n_predicted_avergaing = 800

llsFilterParameters = LLSFilterParameters(dataPath=dataPath)
stereoFilterParameters = StereoFilterParameters()
llsfilter = LLSfilter(dataPath=dataPath, llsFilterParameters=llsFilterParameters, stereoFilterParameters=stereoFilterParameters, debuPlots=False)

noise_lims = [-100, 100]
ave_noise_lims = [0, 20]

t, x, y, z = llsfilter.retrieveRawData(dataPath=dataPath, apply2Dfilter=False)
time_origin = t[0]
t = [ti - time_origin for ti in t]

x = [xi * 10 for xi in x]

filtered_time_x, filtered_signal_x, filtered_noise_x = extract_noise_2(x, t, windowSize=21, polynomialOrder=5)

t_predicted, x_predicted, y_predicted, z_predicted = llsfilter.run_simulation(dataPath=dataPath)
t_predicted = t_predicted - time_origin
x_predicted = [xi * 10 for xi in x_predicted] #convert to mm
filtered_predicted_time_x, filtered_predicted_signal_x, filtered_predicted_noise_x = extract_noise_2(x_predicted, t_predicted, windowSize=81, polynomialOrder=5)

average_noise = []
average_predicted_noise = []

for i in range(n_avergaing, len(filtered_signal_x)):
    average_noise.append(np.std(filtered_noise_x[i - n_avergaing:i]))


for i in range(n_predicted_avergaing, len(filtered_predicted_time_x)):
    average_predicted_noise.append(np.std(filtered_predicted_noise_x[i - n_predicted_avergaing:i]))

fig, axs = plt.subplots(3,3, sharex=True)

axs[0, 0].plot(t, x)
axs[0, 0].plot(t_predicted, x_predicted)

axs[0, 1].plot(t, x)
axs[0, 1].plot(filtered_time_x, filtered_signal_x)
ax = axs[0, 1].twinx()
ax.plot(filtered_time_x, filtered_noise_x, color='black')
ax.set_ylim(noise_lims)

axs[1, 1].plot(filtered_time_x[n_avergaing:], average_noise)
axs[1, 1].set_ylim(ave_noise_lims)

axs[0, 2].plot(t_predicted, x_predicted)
ax = axs[0, 2].twinx()
ax.plot(filtered_predicted_time_x, filtered_predicted_noise_x, color='black')
ax.set_ylim(noise_lims)

axs[1, 2].plot(filtered_predicted_time_x[n_predicted_avergaing:], average_predicted_noise)
axs[1, 2].set_ylim(ave_noise_lims)
plt.show()
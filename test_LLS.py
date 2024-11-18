from LLSfilter import LLSFilterParameters, LLSfilter, SR_vector4d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


dataPath = './data/windows_traces/2024-11-15___17_19_25/'
#initialize filter parameterts with the ones in ft_user.ini
llsFilterParameters = LLSFilterParameters(path=dataPath)

llsfilter = LLSfilter(llsFilterParameters)

t, x, y, z = llsfilter.retrieveRawData(dataPath=dataPath)
time_origin = t[0]
t = t - time_origin
raw = [x, y, z]

label = ["x", "y", "z"]

t_predicted, x_predicted, y_predicted, z_predicted = llsfilter.outputThread(dataPath=dataPath)
t_predicted = t_predicted - time_origin
predicted = [x_predicted, y_predicted, z_predicted]

data = pd.read_csv(dataPath + '2024-11-15___17_19_25_predictedWeaving.csv')
t_predicted_etr = data['timeLogged']
#Debugging
t_mostRecent_etr = data[' mostRecentObservationTime']
delay_capture_logged = t_predicted_etr - t_mostRecent_etr

t_predicted_etr = t_predicted_etr - time_origin

x_predicted_etr = 0.5 * (data[' leftEye.x'] + data[' rightEye.x'])
y_predicted_etr = 0.5 * (data[' leftEye.y'] + data[' rightEye.y'])
z_predicted_etr = 0.5 * (data[' leftEye.z'] + data[' rightEye.z'])
predicted_etr = [x_predicted_etr, y_predicted_etr, z_predicted_etr]

#debug data
currentTime, lastDataTimestamps = llsfilter.simulatorDebugData(dataPath=dataPath)
delay_capture_current = currentTime - lastDataTimestamps
currentTime = currentTime - time_origin

print(np.average(delay_capture_logged))

for i in range(0, 3):
    plt.subplot(5, 1, i + 3)
    plt.plot(t, raw[i], '-o', color='red')
    plt.plot(t_predicted, predicted[i], '-o', color='blue' )
    plt.plot(t_predicted_etr, predicted_etr[i], '-o', color='green' )
    plt.legend(['Raw ' + label[i], 'Predicted ' + label[i], 'Predicted in EyeTracker' + label[i]])
    plt.ylabel("Signal, signal units")
    plt.xlabel("time, s")

plt.subplot(5, 1, 1)
plt.plot(t_predicted_etr, delay_capture_logged, '-o')
plt.plot(currentTime, delay_capture_current, '-o')

plt.subplot(5,1,2)
plt.plot(np.diff(t_predicted_etr), '-o')
plt.plot(np.diff(currentTime), '-o')
plt.show()
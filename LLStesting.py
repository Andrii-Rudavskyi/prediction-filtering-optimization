from LLSfilter import LLSfilter, LLSFilterParameters, FilterType
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



dataID = '2024-12-02___14_24_31'
#dataID = '2024-12-02___14_24_38'
#dataID = '2024-12-02___14_24_43'
#dataID = '2024-12-02___14_24_51'

dataPath = './data/windows_traces/'

#dataPath= 'E:/git/1/TobiiIntegration/EyeTracker/build/Recordings/'

dataPath = dataPath + dataID + '/'
#initialize filter parameterts with the ones in ft_user.ini
llsFilterParameters = LLSFilterParameters(dataPath=dataPath, filterType=FilterType.WeavingPoseFilter)

llsfilter = LLSfilter(llsFilterParameters, debuPlots=False)

t, x, y, z = llsfilter.retrieveRawData(dataPath=dataPath)
time_origin = t[0]
t = t - time_origin
raw = [x, y, z]

label = ["x", "y", "z"]

t_predicted, x_predicted, y_predicted, z_predicted = llsfilter.outputThread(dataPath=dataPath)
t_predicted = t_predicted - time_origin
predicted = [x_predicted, y_predicted, z_predicted]

if (llsFilterParameters.filterType == FilterType.WeavingPoseFilter):
    data = pd.read_csv(dataPath + dataID + '_filter_data_weaving.csv')
else:
    data = pd.read_csv(dataPath + dataID + '_filter_data_lookaroud.csv')
t_predicted_etr = data['current_time']
#t_predicted_etr = data[' usedInterpolatedTime']
#Debugging

t_predicted_etr = t_predicted_etr + llsFilterParameters.predictionTime

t_predicted_etr = t_predicted_etr - time_origin

x_predicted_etr = data['predicted_x']
y_predicted_etr = data['predicted_y']
z_predicted_etr = data['predicted_z']
predicted_etr = [x_predicted_etr, y_predicted_etr, z_predicted_etr]

#debug data
currentTime, lastDataTimestamps = llsfilter.simulatorDebugData(dataPath=dataPath)
delay_capture_current = currentTime - lastDataTimestamps
currentTime = currentTime - time_origin


for i in range(0, 3):
    plt.subplot(5, 1, i + 3)
    plt.plot(t, raw[i], '-o', color='red')
    plt.plot(t_predicted, predicted[i], '-o', color='blue' )
    plt.plot(t_predicted_etr, predicted_etr[i], '-o', color='green' )
    plt.legend(['Raw ' + label[i], 'Predicted ' + label[i], 'Predicted in EyeTracker' + label[i]])
    plt.ylabel("Signal, signal units")
    plt.xlabel("time, s")

plt.subplot(5, 1, 1)
plt.plot(currentTime, delay_capture_current, '-o')

plt.subplot(5,1,2)
plt.plot(np.diff(t_predicted_etr), '-o')
plt.plot(np.diff(currentTime), '-o')
plt.show()

print('Average delay_capture_current', np.average(delay_capture_current))

print('Total latency = ', np.average(delay_capture_current) + llsFilterParameters.predictionTime)
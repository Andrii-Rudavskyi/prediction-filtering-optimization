import matplotlib.pyplot as plt
from LLSfilter import LLSfilter, LLSFilterParameters, FilterType
import numpy as np

data_OFF = './data/2D_filter_influence_investigation/OFF/'
data_ON = './data/2D_filter_influence_investigation/ON/'

# Data arranged in the order of increasing velocity
OFF_data_list = ['2025-01-23___11_59_22', '2025-01-23___12_46_54', '2025-01-23___13_11_59']
ON_data_list = ['2025-01-23___12_01_25', '2025-01-23___13_08_19', '2025-01-23___13_15_54']

fig, axs = plt.subplots(3)

plots_axs1 = []
plots_axs2 = []

for i in range(0, 3):
    dataPath_filter_OFF = data_OFF + OFF_data_list[i] + '/'
    dataPath_filter_ON = data_ON + ON_data_list[i] + '/'

    llsFilterParameters_OFF = LLSFilterParameters(dataPath=dataPath_filter_OFF, filterType=FilterType.WeavingPoseFilter)
    llsfilter_OFF = LLSfilter(llsFilterParameters_OFF, debuPlots=False)

    llsFilterParameters_ON = LLSFilterParameters(dataPath=dataPath_filter_ON, filterType=FilterType.WeavingPoseFilter)
    llsfilter_ON = LLSfilter(llsFilterParameters_ON, debuPlots=False)

    t, x, y, z = llsfilter_OFF.retrieveRawData(dataPath=dataPath_filter_OFF)
    time_origin_OFF = t[0]
    t = [ti - time_origin_OFF for ti in t]
    x = [xi * 10 for xi in x]  # convert to mm

    x3D_2D_filter_OFF = x
    t_2D_filter_OFF = t

    t, x, y, z = llsfilter_ON.retrieveRawData(dataPath=dataPath_filter_ON)
    time_origin_ON = t[0]
    t = [ti - time_origin_ON for ti in t]
    x = [xi * 10 for xi in x]  # convert to mm

    x3D_2D_filter_ON = x
    t_2D_filter_ON = t

    velocity = np.diff(x3D_2D_filter_ON) / np.diff(t_2D_filter_ON)

    t_predicted, x_predicted, y_predicted, z_predicted = llsfilter_OFF.outputThread(dataPath=dataPath_filter_OFF)
    t_predicted = t_predicted - time_origin_OFF
    x_predicted = [xi * 10 for xi in x_predicted]  # convert to mm

    t_predicted_2D_filter_OFF = t_predicted
    x3D_predicted_2D_filter_OFF = x_predicted

    t_predicted, x_predicted, y_predicted, z_predicted = llsfilter_ON.outputThread(dataPath=dataPath_filter_ON)
    t_predicted = t_predicted - time_origin_ON
    x_predicted = [xi * 10 for xi in x_predicted]  # convert to mm

    t_predicted_2D_filter_ON = t_predicted
    x3D_predicted_2D_filter_ON = x_predicted

    ax2 = axs[i].twinx()

    pl = axs[i].plot(t_2D_filter_OFF, x3D_2D_filter_OFF)
    axs[i].plot(t_2D_filter_ON, x3D_2D_filter_ON)
    axs[i].plot(t_predicted_2D_filter_OFF, x3D_predicted_2D_filter_OFF)
    axs[i].plot(t_predicted_2D_filter_ON, x3D_predicted_2D_filter_ON)
    axs[i].set_ylabel('3D X, mm')
    axs[i].set_xlabel('Time, s')

    plots_axs1.append(pl)

    pl2 = ax2.plot(t_2D_filter_ON[1:len(t_2D_filter_ON)], velocity, '--', color='black')
    ax2.set_ylabel('Velocity, mm/s')
    plots_axs2.append(pl2)

fig.legend(plots_axs1, labels=['X 3D before prediction, 2D filter OFF', 'X 3D before prediction, 2D filter ON',
                               'X 3D after prediction, 2D filter OFF', 'X 3D after prediction, 2D filter ON'], loc='upper left')
plt.show()
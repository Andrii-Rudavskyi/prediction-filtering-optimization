from LLSfilter import LLSfilter, LLSFilterParameters, FilterType, PolynomialFilterParameters, calculate_prediction_error
import matplotlib.pyplot as plt
import numpy as np

dataID = '2024-12-02___14_24_31'
#dataID = '2024-12-02___14_24_38'
#dataID = '2024-12-02___14_24_43'
#dataID = '2024-12-02___14_24_51'

dataPath = './data/windows_traces/'
dataPath = dataPath + dataID + '/'

#-----------------Filter 1-------------------------------------------------------
llsFilterParameters1 = LLSFilterParameters(dataPath=dataPath, filterType=FilterType.WeavingPoseFilter, usePrediction=True)
llsfilter1 = LLSfilter(llsFilterParameters1, debuPlots=False)

t, x, y, z = llsfilter1.retrieveRawData(dataPath=dataPath)
time_origin = t[0]
t = t - time_origin
raw = [x, y, z]

label = ["x", "y", "z"]

t_predicted, x_predicted, y_predicted, z_predicted = llsfilter1.outputThread(dataPath=dataPath)
t_predicted = t_predicted - time_origin
predicted_filter_1 = [x_predicted, y_predicted, z_predicted]

predicted_error_x = calculate_prediction_error(t_predicted, x_predicted, t, x)
predicted_error_y = calculate_prediction_error(t_predicted, y_predicted, t, y)
predicted_error_z = calculate_prediction_error(t_predicted, z_predicted, t, z)

error_filter_1 = [predicted_error_x, predicted_error_y, predicted_error_z]
#---------------------------------------------------------------------------------

# Filter 2------------------------------------------------------------------------
llsFilterParameters2 = LLSFilterParameters(dataPath=dataPath, filterType=FilterType.PolynomialFit, predictionTime=0.04, usePrediction=True, polynomialFilterParameters=PolynomialFilterParameters(n_buffers=np.array([4, 6, 12]), polynomialOrder=2))
llsfilter2 = LLSfilter(llsFilterParameters2, debuPlots=False)

t_predicted, x_predicted, y_predicted, z_predicted = llsfilter2.outputThread(dataPath=dataPath)
t_predicted = t_predicted - time_origin
predicted_filter_2 = [x_predicted, y_predicted, z_predicted]

predicted_error_x = calculate_prediction_error(t_predicted, x_predicted, t, x)
predicted_error_y = calculate_prediction_error(t_predicted, y_predicted, t, y)
predicted_error_z = calculate_prediction_error(t_predicted, z_predicted, t, z)

error_filter_2 = [predicted_error_x, predicted_error_y, predicted_error_z]
#---------------------------------------------------------------------------------



for i in range(0, 3):
    plt.subplot(3, 2, 2 * i + 1)
    plt.plot(t, raw[i], '-o', color='red', markersize=2, linewidth=1)
    plt.plot(t_predicted, predicted_filter_1[i], '-o', color='blue', markersize=2, linewidth=1)
    plt.plot(t_predicted, predicted_filter_2[i], '-o', color='green', markersize=2, linewidth=1)
    #plt.plot(t_predicted_etr, predicted_etr[i], '-o', color='green' )
    plt.legend(['Raw ' + label[i], 'Filter 1, predicted ' + label[i], 'Filter 2, predicted ' + label[i] ])
    plt.ylabel("Signal, signal units")
    plt.xlabel("time, s")

    plt.subplot(3, 2, 2 * i + 2)
    plt.plot(t_predicted, error_filter_1[i], '-o', color='blue', markersize=2, linewidth=1)
    plt.plot(t_predicted, error_filter_2[i], '-o', color='green', markersize=2, linewidth=1)
    plt.legend(['Filter 1, predicted error ' + label[i], 'Filter 2, predicted error ' + label[i]])
    plt.ylim([-1, 1])

plt.show()
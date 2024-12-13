import numpy as np
import pandas as pd
import os
import configparser
from importCalibration import cameraIntrinsics, Extrinsics, stereoCameraCalibrationData, importCalibration
from enum import Enum
import matplotlib.pyplot as plt

class RawDataType(Enum):
    WINDOWS = 0
    ANDROID = 1

class FilterType(Enum):
    WeavingPoseFilter = 0
    LookaroundFilter = 1
    LastVelosity = 3
    PolynomialFit = 4

class PolynomialFilterParameters:
    def __init__(self, polynomialOrder = None, n_buffers = None, filterInAngualarSpace = None):
        self.polynomialOrder = polynomialOrder if polynomialOrder is not None else 2
        self.n_buffers = n_buffers if n_buffers is not None else np.array([3, 3, 3])
        self.filterInAngualarSpace = filterInAngualarSpace if filterInAngualarSpace is not None else False

class Output:
    def __init__(self):
        self.predicted_t = []
        self.predicted_x = []
        self.predicted_y = []
        self.predicted_z = []

class DebugData:
    def __init__(self):
        self.h_x_prev = []
        self.h_y_prev = []
        self.h_z_prev = []
        self.h_t_prev = []
        self.pred_x = []
        self.pred_y = []
        self.pred_z = []
        self.pred_t = []
        self.velocity_limited_x = []
        self.velocity_limited_y = []
        self.velocity_limited_z = []
        self.noise_rejection_x = []
        self.noise_rejection_y = []
        self.noise_rejection_z = []


class SR_vector4d:
    def __init__(self, x=0, y=0, z=0, t=0):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

class SR_vector3d:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

class SpeedLimitFilter:
    def __init__(self, speedLimit_cm_s = SR_vector3d(x=600, y=600, z=300)):
        self.speedLimit_cm_s = speedLimit_cm_s
        self.prevOutput = SR_vector3d(x=0, y=0, z=0)
        self.prevTime = -1

    def filter(self, input = SR_vector3d(x=0, y=0, z=0), time_s = 0):
        #limit the speed of the filtered output
        output = input
        v = SR_vector3d(x=0, y=0, z=0)

        #If prevTime is a valid value
        if (self.prevTime > 0):
            dt = time_s - self.prevTime
            if (dt > 0):
                v.x = (output.x - self.prevOutput.x) / dt
                v.y = (output.y - self.prevOutput.y) / dt
                v.z = (output.z - self.prevOutput.z) / dt

                if (abs(v.x) > self.speedLimit_cm_s.x):
                    v.x = np.sign(v.x) * self.speedLimit_cm_s.x
                    print("Speed limit filter working on X")
                if (abs(v.y > self.speedLimit_cm_s.y)):
                    v.y = np.sign(v.y) * self.speedLimit_cm_s.y
                    print("Speed limit filter working on Y")
                if (abs(v.z > self.speedLimit_cm_s.z)):
                    v.z = np.sign(v.z) * self.speedLimit_cm_s.z
                    print("Speed limit filter working on Z")

                output.x = self.prevOutput.x + v.x * dt
                output.y = self.prevOutput.y + v.y * dt
                output.z = self.prevOutput.z + v.z * dt

        self.prevTime = time_s
        self.prevOutput = output

        return output

class ExponentialDecayFilter:
    def __init__(self, exponentialDecayAlpha = SR_vector3d(x=0.1, y=0.1, z=0.05)):
        self.exponentialDecayAlpha = exponentialDecayAlpha
        self.prevOutput = SR_vector3d(x=0.0, y=0.0, z=0.0)

    def filter(self, input):
        output = SR_vector3d(x=0, y=0, z=0)

        output.x = self.prevOutput.x + (input.x - self.prevOutput.x) * self.exponentialDecayAlpha.x
        output.y = self.prevOutput.y + (input.y - self.prevOutput.y) * self.exponentialDecayAlpha.y
        output.z = self.prevOutput.z + (input.z - self.prevOutput.z) * self.exponentialDecayAlpha.z

        self.prevOutput = output

        return output

class NoiseRejectionParameters:
    def __init__(self, noiseRejectionAlpha = SR_vector3d(x=0.1, y=0.1, z=0.1), noiseRejectionThreshold = SR_vector3d(x=1, y=1, z=1),
                 noiseRejectionThresholdSpeedRange = SR_vector3d(x=5,y=5, z=5), noiseRejectionThresholdAlpha = SR_vector3d(x=0.01, y=0.01, z=0.01)):
        self.noiseRejectionAlpha = noiseRejectionAlpha
        self.noiseRejectionThreshold = noiseRejectionThreshold
        self.noiseRejectionThresholdSpeedRange = noiseRejectionThresholdSpeedRange
        self.noiseRejectionThresholdAlpha = noiseRejectionThresholdAlpha

class NoiseRejectionFilter:
    def __init__(self, parameters = NoiseRejectionParameters()):
        self.parameters = parameters
        self.threshold = SR_vector3d(x=0, y=0, z=0)
        self.prevOutput = SR_vector3d(x=0, y=0, z=0)

    def filter(self, input = SR_vector3d(x=0, y=0, z=0), velocity_cm_s = SR_vector3d(x=0, y=0, z=0)):

        diff = SR_vector3d(x=0, y=0, z=0)
        output_smoothed = SR_vector3d(x=0, y=0, z=0)

        diff.x = input.x - self.prevOutput.x
        diff.y = input.y - self.prevOutput.y
        diff.z = input.z - self.prevOutput.z

        output_smoothed.x = self.prevOutput.x + diff.x * self.parameters.noiseRejectionAlpha.x
        output_smoothed.y = self.prevOutput.y + diff.y * self.parameters.noiseRejectionAlpha.y
        output_smoothed.z = self.prevOutput.z + diff.z * self.parameters.noiseRejectionAlpha.z

        self.threshold.x = self.threshold.x + (self.parameters.noiseRejectionThreshold.x - self.threshold.x) * self.parameters.noiseRejectionThresholdAlpha.x
        self.threshold.y = self.threshold.y + (self.parameters.noiseRejectionThreshold.y - self.threshold.y) * self.parameters.noiseRejectionThresholdAlpha.y
        self.threshold.z = self.threshold.z + (self.parameters.noiseRejectionThreshold.z - self.threshold.z) * self.parameters.noiseRejectionThresholdAlpha.z

        if (abs(velocity_cm_s.x) > self.parameters.noiseRejectionThresholdSpeedRange.x):
            self.threshold.x = 0
        if (abs(velocity_cm_s.y) > self.parameters.noiseRejectionThresholdSpeedRange.y):
            self.threshold.y = 0
        if (abs(velocity_cm_s.z) > self.parameters.noiseRejectionThresholdSpeedRange.z):
            self.threshold.z = 0

        output = self.prevOutput

        if (abs(diff.x) > self.threshold.x):
            output.x += np.sign(diff.x) * (abs(diff.x) - self.threshold.x)
        else:
            output.x = output_smoothed.x

        if (abs(diff.y) > self.threshold.y):
            output.y += np.sign(diff.y) * (abs(diff.y) - self.threshold.y)
        else:
            output.y = output_smoothed.y

        if (abs(diff.z) > self.threshold.z):
            output.z += np.sign(diff.z) * (abs(diff.z) - self.threshold.z)
        else:
            output.z = output_smoothed.z

        self.prevOutput = output

        return output

class LLSFilterParameters:
    def __init__(self, cameraLatency_s = None,
                 maxSizeHistory = None,
                 smoothVscale = None,
                 numOutliers = None,
                 useFixedZ = None,
                 fixedZ = None,
                 predictionTime = None,
                 dataPath = None,
                 filterType = FilterType.WeavingPoseFilter,
                 speedLimit_cm_s = None,
                 usePrediction = None,
                 predictZ = None,
                 maxPredictionDistance_cm = None,
                 useExponentialDecay = None,
                 exponentialDecayAlpha = None,
                 useNoiseRejection = None,
                 noiseRejectionAlpha = None,
                 noiseRejectionThreshold_cm = None,
                 noiseRejectionThresholdSpeedRange_cm_s = None,
                 noiseRejectionThresholdAlpha = None,
                 polynomialFilterParameters = None
                 ):
        self.cameraLatency_s = cameraLatency_s if cameraLatency_s is not None else 0.02436
        self.maxSizeHistory = maxSizeHistory if maxSizeHistory is not None else np.array([4, 6, 12])
        self.dataPath = dataPath if dataPath is not None else ''
        self.smoothVscale = smoothVscale if smoothVscale is not None else np.array([0, 0, 0])
        self.numOutliers = numOutliers if numOutliers is not None else np.array([0, 0, 0])
        self.useFixedZ = useFixedZ if useFixedZ is not None else False
        self.fixedZ = fixedZ if fixedZ is not None else 60
        self.predictionTime = predictionTime if predictionTime is not None else 0.04
        self.filterType = filterType if filterType is not None else FilterType.WeavingPoseFilter
        self.speedLimit_cm_s = speedLimit_cm_s if speedLimit_cm_s is not None else np.array([600, 600, 300])
        self.usePrediction = usePrediction if usePrediction is not None else True
        self.predictZ = predictZ if predictZ is not None else True
        self.maxPredictionDistance_cm = maxPredictionDistance_cm if maxPredictionDistance_cm is not None else 15
        self.useExponentialDecay = useExponentialDecay if useExponentialDecay is not None else False
        self.exponentialDecayAlpha = exponentialDecayAlpha if exponentialDecayAlpha is not None else np.array([0.1, 0.1, 0.05])
        self.useNoiseRejection = useNoiseRejection if useNoiseRejection is not None else True
        self.noiseRejectionAlpha = noiseRejectionAlpha if noiseRejectionAlpha is not None else np.array([0.1, 0.1, 0.1])
        self.noiseRejectionThreshold_cm = noiseRejectionThreshold_cm if noiseRejectionThreshold_cm is not None else np.array([1.0, 1.0, 1.0])
        self.noiseRejectionThresholdSpeedRange_cm_s = noiseRejectionThresholdSpeedRange_cm_s if noiseRejectionThresholdSpeedRange_cm_s is not None else np.array([5, 5, 5])
        self.noiseRejectionThresholdAlpha = noiseRejectionThresholdAlpha if noiseRejectionThresholdAlpha is not None else np.array([0.01, 0.01, 0.01])
        self.polynomialFilterParameters = polynomialFilterParameters if polynomialFilterParameters is not None else PolynomialFilterParameters()
        if (self.useFixedZ):
            print("fixedZ", self.fixedZ)
        if (len(self.dataPath) > 0):
            print("Importing filtering parameters from" + self.dataPath)

            print("The following parameters have been changed from ft_user.ini:")

            config = configparser.ConfigParser(inline_comment_prefixes=';')
            config.sections()

            config.read(self.dataPath + 'resources/ft_user.ini')
            config.sections()

            if cameraLatency_s is None:
                self.cameraLatency_s = config.getfloat('ApplicationParameters', 'cameraLatency_s')
            else:
                print('cameraLatency_s', self.cameraLatency_s)

            if predictionTime is None:
                if self.filterType == FilterType.WeavingPoseFilter:
                    self.predictionTime = config.getfloat('ApplicationParameters', 'predictionTime_s')
                else:
                    self.predictionTime = config.getfloat('ApplicationParameters', 'predictionTimeScene_s')
            else:
                print('predictionTime: ', self.predictionTime)

            sectionName = None
            if (filterType == FilterType.WeavingPoseFilter):
                sectionName = 'WeavingPoseFilter'
            elif (filterType == FilterType.LookaroundFilter):
                sectionName = 'LookaroundFilter'
            print('Loading parameters for: ', sectionName)

            if sectionName is not None:
                if smoothVscale is None:
                    smoothVscale_str_vec = config.get(sectionName, 'smoothVscale').strip('[').strip(']').split(',')
                    for i in range(0, 3):
                        self.smoothVscale[i] = float(smoothVscale_str_vec[i])
                else:
                    print("smoothVScale", self.smoothVscale)

                if maxSizeHistory is None:
                    filterHistory_str_vec = config.get(sectionName, 'filterHistory').strip('[').strip(']').split(
                        ',')
                    for i in range(0, 3):
                        self.maxSizeHistory[i] = int(filterHistory_str_vec[i])

                if numOutliers is None:
                    numOutliers_str_vec = config.get(sectionName, 'numOutliers').strip('[').strip(']').split(',')
                    for i in range(0, 3):
                        self.numOutliers[i] = int(numOutliers_str_vec[i])

                if speedLimit_cm_s is None:
                    speedLimit_cm_s_str_vec = config.get(sectionName, 'speedLimit_cm_s').strip('[').strip(']').split(',')
                    for i in range(0, 3):
                        self.speedLimit_cm_s[i] = float(speedLimit_cm_s_str_vec[i])
                else:
                    print('speedLimit_cm_s: ', self.speedLimit_cm_s)

                if usePrediction is None:
                    self.usePrediction = config.getboolean(sectionName, 'usePrediction')
                else:
                    print('usePrediction ', self.usePrediction)

                if predictZ is None:
                    try:
                        self.predictZ = config.getboolean(sectionName, 'predictZ')
                    except configparser.Error:
                        print("predictZ is not specified, default will be used: ", self.predictZ)
                else:
                    print('predictZ', self.predictZ)

                if useFixedZ is None:
                    try:
                        self.useFixedZ = config.getboolean(sectionName, 'useFixedZ')
                    except configparser.Error:
                        print("useFixedZ is not specified, default will be used: ", self.useFixedZ)
                else:
                    print('useFixedZ ', useFixedZ)

                if fixedZ is None:
                    if self.useFixedZ:
                        self.fixedZ = config.getfloat(sectionName, 'fixedZ_cm')
                else:
                    print('fixedZ_cm: ', self.fixedZ)

                if maxPredictionDistance_cm is None:
                    self.maxPredictionDistance_cm = config.getfloat(sectionName, 'maxPredictionDistance_cm')
                else:
                    print('maxPredictionDistance_cm', self.maxPredictionDistance_cm)


                if useExponentialDecay is None:
                    self.useExponentialDecay = config.getboolean(sectionName, 'useExponentialDecay')
                else:
                    print('useExponentialDecay', self.useExponentialDecay)

                if useNoiseRejection is None:
                    self.useNoiseRejection = config.getboolean(sectionName, 'useNoiseRejection')
                else:
                    print('useNoiseRejection', self.useNoiseRejection)


                if (self.useExponentialDecay):
                    if exponentialDecayAlpha is None:
                        exponentialDecayAlpha_str_vec = config.get(sectionName, 'exponentialDecayAlpha').strip('[').strip(']').split(',')
                        for i in range(0, 3):
                            self.exponentialDecayAlpha[i] = float(exponentialDecayAlpha_str_vec[i])
                    else:
                        print('exponentialDecayAlpha: ', self.exponentialDecayAlpha)

                if noiseRejectionAlpha is None:
                    noiseRejectionAlpha_str_vec = config.get(sectionName, 'noiseRejectionAlpha').strip('[').strip(']').split(',')
                    for i in range(0, 3):
                        self.noiseRejectionAlpha[i] = float(noiseRejectionAlpha_str_vec[i])
                else:
                    print('noiseRejectionAlpha: ', self.noiseRejectionAlpha)


                if noiseRejectionThreshold_cm is None:
                    noiseRejectionThreshold_cm_str_vec = config.get(sectionName, 'noiseRejectionThreshold_cm').strip('[').strip(']').split(',')
                    for i in range(0, 3):
                        self.noiseRejectionThreshold_cm[i] = float(noiseRejectionThreshold_cm_str_vec[i])
                else:
                    print('noiseRejectionThreshold_cm', self.noiseRejectionThreshold_cm)

                if noiseRejectionThresholdSpeedRange_cm_s is None:
                    noiseRejectionThresholdSpeedRange_cm_s_str_vec = config.get(sectionName, 'noiseRejectionThresholdSpeedRange_cm_s').strip('[').strip(']').split(',')
                    for i in range(0, 3):
                        self.noiseRejectionThresholdSpeedRange_cm_s[i] = float(noiseRejectionThresholdSpeedRange_cm_s_str_vec[i])
                else:
                    print('noiseRejectionThresholdSpeedRange_cm_s: ', self.noiseRejectionThresholdSpeedRange_cm_s)

                if noiseRejectionThresholdAlpha is None:
                    noiseRejectionThresholdAlpha_str_vec = config.get(sectionName,'noiseRejectionThresholdAlpha').strip('[').strip(']').split(',')
                    for i in range(0, 3):
                        self.noiseRejectionThresholdAlpha[i] = float(noiseRejectionThresholdAlpha_str_vec[i])
                else:
                    print('noiseRejectionThresholdAlpha: ', self.noiseRejectionThresholdAlpha)

        print("cameraLatency_s", self.cameraLatency_s)
        if self.filterType is FilterType.WeavingPoseFilter or FilterType.LookaroundFilter:
            print("--------------------FILTERING PARAMETERS------------------------------------------")
            print("maxSizeHistory", self.maxSizeHistory)
            print("smoothVscale", self.smoothVscale)
            print("numOutliers", self.numOutliers)
            print("predictionTime", self.predictionTime)
            print("speedLimit_cm_s", self.speedLimit_cm_s)
            print("usePrediction ", self.usePrediction)
            print("predictZ ", self.predictZ)
            print("useFizedZ", self.useFixedZ)
            print("FixedZ ", self.fixedZ)
            print("maxPredictionDistance_cm ", self.maxPredictionDistance_cm)
            print("useExponentialDecay ", self.useExponentialDecay)
            print("exponentialDecayAlpha ", self.exponentialDecayAlpha)
            print("useNoiseRejection ", self.useNoiseRejection)
            print("noiseRejectionAlpha ", self.noiseRejectionAlpha)
            print("noiseRejectionThreshold_cm ", self.noiseRejectionThreshold_cm)
            print("noiseRejectionThresholdSpeedRange_cm_s ", self.noiseRejectionThresholdSpeedRange_cm_s)
            print("noiseRejectionThresholdAlpha ", self.noiseRejectionThresholdAlpha)
        elif self.filterType is FilterType.PolynomialFit:
            print("n_buffers: ", self.polynomialFilterParameters.n_buffers)
            print("polynomialOrder: ", self.polynomialFilterParameters.polynomialOrder)

class LLSfilter:
    def __init__(self, filterParameters = LLSFilterParameters(), calibrationData = stereoCameraCalibrationData(), debuPlots = False):
        self.filterParameters = filterParameters
        self.calibrationData = calibrationData
        self.speedLimitFilter = SpeedLimitFilter(speedLimit_cm_s=SR_vector3d(x=self.filterParameters.speedLimit_cm_s[0],
                                                                             y=self.filterParameters.speedLimit_cm_s[1],
                                                                             z=self.filterParameters.speedLimit_cm_s[2]))
        self.exponentialDecayFilter = ExponentialDecayFilter(exponentialDecayAlpha=SR_vector3d(x=self.filterParameters.exponentialDecayAlpha[0],
                                                                                               y=self.filterParameters.exponentialDecayAlpha[1],
                                                                                               z=self.filterParameters.exponentialDecayAlpha[2]))
        self.noiseRejectionFilter = NoiseRejectionFilter(parameters=NoiseRejectionParameters(noiseRejectionAlpha=SR_vector3d(x=self.filterParameters.noiseRejectionAlpha[0], y=self.filterParameters.noiseRejectionAlpha[1], z=self.filterParameters.noiseRejectionAlpha[2]),
                                                                                             noiseRejectionThreshold=SR_vector3d(x=self.filterParameters.noiseRejectionThreshold_cm[0], y=self.filterParameters.noiseRejectionThreshold_cm[1], z=self.filterParameters.noiseRejectionThreshold_cm[2]),
                                                                                             noiseRejectionThresholdSpeedRange=SR_vector3d(x=self.filterParameters.noiseRejectionThresholdSpeedRange_cm_s[0], y=self.filterParameters.noiseRejectionThresholdSpeedRange_cm_s[1], z=self.filterParameters.noiseRejectionThresholdSpeedRange_cm_s[2]),
                                                                                             noiseRejectionThresholdAlpha=SR_vector3d(x=self.filterParameters.noiseRejectionThresholdAlpha[0], y=self.filterParameters.noiseRejectionThresholdAlpha[1], z=self.filterParameters.noiseRejectionThresholdAlpha[2])))
        self.debugPlots = debuPlots

    def LLS2(self, xvals, yvals, xpredict):
        XSx = 0
        XSy = 0
        XSxy = 0
        XSxx = 0
        XSxxx = 0
        XSxxy = 0
        XSxxxx = 0

        n = len(yvals)
        n_inverse = 1.0 / n

        for ii in range(0, n):
            XSx += xvals[ii]
            XSy += yvals[ii]
            XSxy += xvals[ii] * yvals[ii]
            XSxx += xvals[ii] * xvals[ii]

            XSxxx += xvals[ii] * xvals[ii] * xvals[ii]
            XSxxy += xvals[ii] * xvals[ii] * yvals[ii]
            XSxxxx += xvals[ii] * xvals[ii] * xvals[ii] * xvals[ii]

        Sxx = XSxx - XSx * XSx * n_inverse
        Sxy = XSxy - XSx * XSy * n_inverse
        Sxx2 = XSxxx - XSx * XSxx * n_inverse
        Sx2y = XSxxy - XSxx * XSy * n_inverse
        Sx2x2 = XSxxxx - XSxx * XSxx * n_inverse

        d = Sxx * Sx2x2 - Sxx2 * Sxx2
        d_inverse = 1 / d

        a = (Sx2y * Sxx - Sxy * Sxx2) * d_inverse
        b = (Sxy * Sx2x2 - Sx2y * Sxx2) * d_inverse
        c = (XSy - b * XSx - a * XSxx) * n_inverse

        v = 2 * a * xpredict + b

        predicted = a * xpredict * xpredict + b * xpredict + c

        return v, predicted

    def genericFilterWithHistory(self, history, currentTime):
        #history is a pair of timestamp and coordinate

        #Values to use for curve fitting
        xs = []
        ys = []
        zs = []

        # Times to use for curve fitting
        timesX = []
        timesY = []
        timesZ = []

        # Each dimension mightb include a different range of the history
        istartX = max(0, len(history) - self.filterParameters.maxSizeHistory[0])
        istartY = max(0, len(history) - self.filterParameters.maxSizeHistory[1])
        istartZ = max(0, len(history) - self.filterParameters.maxSizeHistory[2])

        #Pivot
        imid = int((istartX + len(history)) / 2)
        pivotTime = history[imid].t # pivot time in the middle of history

        for ii in range(0, len(history)):
            if (ii >= istartX):
                timesX.append(history[ii].t - pivotTime)
            if (ii >= istartY):
                timesY.append(history[ii].t - pivotTime)
            if (ii >= istartZ):
                timesZ.append(history[ii].t - pivotTime)

            x = history[ii].x
            y = history[ii].y
            z = history[ii].z

            if (self.filterParameters.useFixedZ):
                x = (x / z) * self.filterParameters.fixedZ
                y = (y / z) * self.filterParameters.fixedZ

            if (ii >= istartX):
                xs.append(x)

            if (ii >= istartY):
                ys.append(y)

            if (ii >= istartZ):
                zs.append(z)

        #Calculate velocities in the second half of the history
        vx, predictedX = self.LLS2(timesX, xs, (history[-1].t - pivotTime) / 2)
        vy, predictedY = self.LLS2(timesY, ys, (history[-1].t - pivotTime) / 2)
        vz, predictedZ = self.LLS2(timesZ, zs, (history[-1].t - pivotTime) / 2)

        # Calculate velocities in the first half of the history
        vx2, predictedX = self.LLS2(timesX, xs, (history[0].t - pivotTime) / 2)
        vy2, predictedY = self.LLS2(timesY, ys, (history[0].t - pivotTime) / 2)
        vz2, predictedZ = self.LLS2(timesZ, zs, (history[0].t - pivotTime) / 2)

        if (self.filterParameters.useFixedZ):
            vx = vx * vz / self.filterParameters.fixedZ
            vy = vy * vz / self.filterParameters.fixedZ
            vx2 = vx2 * vz2 / self.filterParameters.fixedZ
            vy2 = vy2 * vz2 / self.filterParameters.fixedZ

        vx = max(abs(vx), abs(vx2))
        vy = max(abs(vy), abs(vy2))
        vz = max(abs(vz), abs(vz2))

        sx = 1
        sy = 1
        sz = 1

        if (self.filterParameters.smoothVscale[0] != 0):
            unbound_sx = np.sqrt(abs(vx)) / self.filterParameters.smoothVscale[0]
            sx = max(0, min(1, unbound_sx))
        if (self.filterParameters.smoothVscale[1] != 0):
            unbound_sy = np.sqrt(abs(vy)) / self.filterParameters.smoothVscale[1]
            sy = max(0, min(1, unbound_sy))
        if (self.filterParameters.smoothVscale[2] != 0):
            unbound_sz = np.sqrt(abs(vz)) / self.filterParameters.smoothVscale[2]
            sz = max(0, min(1, unbound_sz))

        # Time with respect to the center of the maintained history
        # This is negative and will result in a smoothing effect

        minPredictionTime = pivotTime - currentTime
        # Time with respect to the last time in history + 0.2s
        # This limits the amount of time that we're trying to predict in the future

        maxPredictionTime = history[-1].t + 0.2 - currentTime;

        # Times to predict for (between minPredictionTime and predictionTime)
        ptimex = minPredictionTime + (self.filterParameters.predictionTime - minPredictionTime) * sx
        ptimey = minPredictionTime + (self.filterParameters.predictionTime - minPredictionTime) * sy
        ptimez = minPredictionTime + (self.filterParameters.predictionTime - minPredictionTime) * sz

        # Reduced to maxPredictionTime if any of the components exceed this limit
        if ptimex > maxPredictionTime:
            ptimex = maxPredictionTime

        if ptimey > maxPredictionTime:
            ptimey = maxPredictionTime

        if ptimez > maxPredictionTime:
            ptimez = maxPredictionTime

        outputData = SR_vector3d()

        vx, predictedX = self.LLS2(timesX, xs, currentTime + ptimex - pivotTime)
        vy, predictedY = self.LLS2(timesY, ys, currentTime + ptimey - pivotTime)
        vz, predictedZ = self.LLS2(timesZ, zs, currentTime + ptimez - pivotTime)

        outputData.x = predictedX
        outputData.y = predictedY
        outputData.z = predictedZ

        if self.filterParameters.useFixedZ:
            outputData.x = outputData.x / self.filterParameters.fixedZ * outputData.z
            outputData.y = outputData.y / self.filterParameters.fixedZ * outputData.z

        return outputData

    def retrieveRawData(self, dataPath):
        x = []
        y = []
        z = []
        t = []

        for file in os.listdir(dataPath):
            if self.filterParameters.filterType == FilterType.WeavingPoseFilter:
                if file.endswith('filter_data_weaving.csv'):
                    filterData = pd.read_csv(dataPath + file)
            else:
                if file.endswith('filter_data_lookaroud.csv'):
                    filterData = pd.read_csv(dataPath + file)

        newDataPointTimestamp = filterData['latest_time'][0]
        x.append(filterData['latest_x'][0])
        y.append(filterData['latest_y'][0])
        z.append(filterData['latest_z'][0])
        t.append(newDataPointTimestamp)

        for j in range(1, len(filterData)):
            if (filterData['latest_time'][j] > newDataPointTimestamp):
                newDataPointTimestamp = filterData['latest_time'][j]
                x.append(filterData['latest_x'][j])
                y.append(filterData['latest_y'][j])
                z.append(filterData['latest_z'][j])
                t.append(newDataPointTimestamp)

        return t, x, y, z

    def simulatorDebugData(self, dataPath):
        for file in os.listdir(dataPath):
            if self.filterParameters.filterType == FilterType.WeavingPoseFilter:
                if file.endswith('filter_data_weaving.csv'):
                    data = pd.read_csv(dataPath + file)
            else:
                if file.endswith('filter_data_lookaroud.csv'):
                    data = pd.read_csv(dataPath + file)

        currentTime = data['current_time']
        lastDataTimestamps = data['latest_time']

        return currentTime, lastDataTimestamps

    def predict1(self, currentTime, history, output):
        # Keep for debugging for now

        if self.debugPlots:
            debugData = DebugData()

        predictedFacePosition = self.genericFilterWithHistory(history=history, currentTime=currentTime)

        # Keep for debugging for now
        if self.debugPlots:
            debugData.pred_x.append(predictedFacePosition.x)
            debugData.pred_y.append(predictedFacePosition.y)
            debugData.pred_z.append(predictedFacePosition.z)
            debugData.pred_t.append(currentTime + self.filterParameters.predictionTime)

        # Prevent change of filtered position of more than 'maxPredictionDistance_cm' in three-dimensional space
        deltaPos = SR_vector3d(x=0, y=0, z=0)
        deltaPos.x = predictedFacePosition.x - history[-1].x
        deltaPos.y = predictedFacePosition.y - history[-1].y
        deltaPos.z = predictedFacePosition.z - history[-1].z

        distanceFromLastMeasurement = np.sqrt(
            deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y + deltaPos.z * deltaPos.z)
        if (distanceFromLastMeasurement > self.filterParameters.maxPredictionDistance_cm):
            deltaPos.x = deltaPos.x * self.filterParameters.maxPredictionDistance_cm / distanceFromLastMeasurement
            deltaPos.y = deltaPos.y * self.filterParameters.maxPredictionDistance_cm / distanceFromLastMeasurement
            deltaPos.z = deltaPos.z * self.filterParameters.maxPredictionDistance_cm / distanceFromLastMeasurement

        predictedFacePosition.x = history[-1].x + deltaPos.x
        predictedFacePosition.y = history[-1].y + deltaPos.y
        predictedFacePosition.z = history[-1].z + deltaPos.z

        outputFacePosition = predictedFacePosition

        # If 'usePrediction' is false, overwrite the output face position with the last face position in the history
        if (not self.filterParameters.usePrediction):
            outputFacePosition.x = history[-1].x
            outputFacePosition.y = history[-1].y
            outputFacePosition.z = history[-1].z
        elif (not self.filterParameters.predictZ):
            # If we predict in x/z and y/z we correct x and y as well, if not we only overwrite z.
            if (self.filterParameters.useFixedZ):
                outputFacePosition.x = (outputFacePosition.x / outputFacePosition.z) * history[-1].z
                outputFacePosition.y = (outputFacePosition.y / outputFacePosition.z) * history[-1].z
                outputFacePosition.z = (outputFacePosition.z / outputFacePosition.z) * history[-1].z
            else:
                outputFacePosition.z = history[-1].z

        # Apply speed limit filter
        outputFacePosition = self.speedLimitFilter.filter(outputFacePosition, currentTime)

        # Keep for debugging for now
        if self.debugPlots:
            debugData.velocity_limited_x.append(outputFacePosition.x)
            debugData.velocity_limited_y.append(outputFacePosition.y)
            debugData.velocity_limited_z.append(outputFacePosition.z)

        # If 'useExponentialDecay' is true, use exponential decay filter
        if (self.filterParameters.useExponentialDecay):
            outputFacePosition = self.exponentialDecayFilter.filter(outputFacePosition)

        # If 'useNoiseRejection' is true, use noise rejection filter
        if (self.filterParameters.useNoiseRejection):
            lastMeasurement = history[-1]
            secondLastMeasurement = history[-2]

            lastMeasuredSpeed = SR_vector3d(x=0, y=0, z=0)
            lastMeasuredSpeed.x = (lastMeasurement.x - secondLastMeasurement.x) / (
                    lastMeasurement.t - secondLastMeasurement.t)
            lastMeasuredSpeed.y = (lastMeasurement.y - secondLastMeasurement.y) / (
                    lastMeasurement.t - secondLastMeasurement.t)
            lastMeasuredSpeed.z = (lastMeasurement.z - secondLastMeasurement.z) / (
                    lastMeasurement.t - secondLastMeasurement.t)

            outputFacePosition = self.noiseRejectionFilter.filter(outputFacePosition, lastMeasuredSpeed)

            # Keep for debugging for now
            if self.debugPlots:
                debugData.noise_rejection_x.append(outputFacePosition.x)
                debugData.noise_rejection_y.append(outputFacePosition.y)
                debugData.noise_rejection_z.append(outputFacePosition.z)

        output.predicted_x.append(outputFacePosition.x)
        output.predicted_y.append(outputFacePosition.y)
        output.predicted_z.append(outputFacePosition.z)
        output.predicted_t.append(currentTime + self.filterParameters.predictionTime)

        # Keep for debugging for now
        h_x = []
        h_y = []
        h_z = []
        h_t = []

        for k in range(0, len(history)):
            h_x.append(history[k].x)
            h_y.append(history[k].y)
            h_z.append(history[k].z)
            h_t.append(history[k].t)

        # Keep for debugging for now
        if self.debugPlots:
            plt.subplot(3, 1, 1)
            plt.plot(h_t, h_x, '-o')
            plt.plot(debugData.h_t_prev, debugData.h_x_prev, '-o')
            plt.plot(debugData.pred_t, debugData.pred_x, '-o', color='black')
            plt.plot(debugData.pred_t, debugData.velocity_limited_x, '-o', color='green')
            plt.plot(debugData.pred_t, debugData.noise_rejection_x, '-o', color='red')
            plt.legend(
                ['New point', 'History', 'Predicted', 'After speed limit filter', 'After noise rejection'])
            plt.axvline(x=currentTime, color='black')

            plt.subplot(3, 1, 2)
            plt.plot(h_t, h_y, '-o')
            plt.plot(debugData.h_t_prev, debugData.h_y_prev, '-o')
            plt.plot(debugData.pred_t, debugData.pred_y, '-o', color='black')
            plt.plot(debugData.pred_t, debugData.velocity_limited_y, '-o', color='green')
            plt.plot(debugData.pred_t, debugData.noise_rejection_y, '-o', color='red')
            plt.legend(
                ['New point', 'History', 'Predicted', 'After speed limit filter', 'After noise rejection'])
            plt.axvline(x=currentTime, color='black')

            plt.subplot(3, 1, 3)
            plt.plot(h_t, h_z, '-o')
            plt.plot(debugData.h_t_prev, debugData.h_z_prev, '-o')
            plt.plot(debugData.pred_t, debugData.pred_z, '-o', color='black')
            plt.plot(debugData.pred_t, debugData.velocity_limited_z, '-o', color='green')
            plt.plot(debugData.pred_t, debugData.noise_rejection_z, '-o', color='red')
            plt.legend(
                ['New point', 'History', 'Predicted', 'After speed limit filter', 'After noise rejection'])
            plt.axvline(x=currentTime, color='black')
            print(currentTime)
            plt.show()

            h_t_prev = h_t
            h_x_prev = h_x
            h_y_prev = h_y
            h_z_prev = h_z

            print("h_t ", len(h_t), h_t)
            print("h_x ", h_x)

    def predict2(self, currentTime, history, output):
        if self.filterParameters.usePrediction:
            t1 = history[-2].t
            t2 = history[-1].t
            x1 = history[-2].x
            x2 = history[-1].x
            y1 = history[-2].y
            y2 = history[-1].y
            z1 = history[-2].z
            z2 = history[-1].z
            vx = (x2 - x1) / (t2 - t1)
            vy = (y2 - y1) / (t2 - t1)
            vz = (z2 - z1) / (t2 - t1)
            x_est = x2 + vx * (currentTime - t2 + self.filterParameters.predictionTime)
            y_est = y2 + vy * (currentTime - t2 + self.filterParameters.predictionTime)
            z_est = z2 + vz * (currentTime - t2 + self.filterParameters.predictionTime)
        else:
            x_est = history[-1].x
            y_est = history[-1].y
            z_est = history[-1].z

        output.predicted_x.append(x_est)
        output.predicted_y.append(y_est)
        output.predicted_z.append(z_est)
        output.predicted_t.append(currentTime + self.filterParameters.predictionTime)

    def predict3(self, currentTime, history, output):
        h_t = []
        h_x = []
        h_y = []
        h_z = []

        for i in range(0, len(history)):
            h_t.append(history[i].t)
            h_x.append(history[i].x)
            h_y.append(history[i].y)
            h_z.append(history[i].z)

        if self.filterParameters.polynomialFilterParameters.filterInAngualarSpace == True:
            #print('x ', x , ' y ', y, ' z ', z )
            h_x = np.divide(h_x, h_z)
            h_y = np.divide(h_y, h_z)

        t_x = h_t[-self.filterParameters.polynomialFilterParameters.n_buffers[0]:]
        x = h_x[-self.filterParameters.polynomialFilterParameters.n_buffers[0]:]
        t_y = h_t[-self.filterParameters.polynomialFilterParameters.n_buffers[1]:]
        y = h_y[-self.filterParameters.polynomialFilterParameters.n_buffers[1]:]
        t_z = h_t[-self.filterParameters.polynomialFilterParameters.n_buffers[2]:]
        z = h_z[-self.filterParameters.polynomialFilterParameters.n_buffers[2]:]

        t_est = currentTime + self.filterParameters.predictionTime

        #For better numerical convergence
        origin = t_x[0]

        t_x = t_x - origin
        t_y = t_y - origin
        t_z = t_z - origin

        p_x = np.polyfit(t_x, x, self.filterParameters.polynomialFilterParameters.polynomialOrder)
        p_y = np.polyfit(t_y, y, self.filterParameters.polynomialFilterParameters.polynomialOrder)
        p_z = np.polyfit(t_z, z, self.filterParameters.polynomialFilterParameters.polynomialOrder)

        x_est = 0
        y_est = 0
        z_est = 0

        for j in range(0, self.filterParameters.polynomialFilterParameters.polynomialOrder + 1):
            x_est = x_est + p_x[j] * np.power(t_est - origin, self.filterParameters.polynomialFilterParameters.polynomialOrder - j)
            y_est = y_est + p_y[j] * np.power(t_est - origin, self.filterParameters.polynomialFilterParameters.polynomialOrder - j)
            z_est = z_est + p_z[j] * np.power(t_est - origin, self.filterParameters.polynomialFilterParameters.polynomialOrder - j)

        if self.filterParameters.polynomialFilterParameters.filterInAngualarSpace == True:
            x_est = x_est * z[-1]
            y_est = y_est * z[-1]

        output.predicted_t.append(t_est)

        if self.filterParameters.usePrediction:
            output.predicted_x.append(x_est)
            output.predicted_y.append(y_est)
            output.predicted_z.append(z_est)
        else:
            output.predicted_x.append(x[-1])
            output.predicted_y.append(y[-1])
            output.predicted_z.append(z[-1])

    def outputThread(self, dataPath):
        maxHistorySize = 0

        if (self.filterParameters.filterType == FilterType.WeavingPoseFilter or self.filterParameters.filterType == FilterType.LookaroundFilter):
            maxHistorySize = max(self.filterParameters.maxSizeHistory)
        elif (self.filterParameters.filterType == FilterType.PolynomialFit):
            maxHistorySize = max(self.filterParameters.polynomialFilterParameters.n_buffers)
        else:
            maxHistorySize = 3

        history = []
        output = Output()

        for file in os.listdir(dataPath):
            if self.filterParameters.filterType == FilterType.WeavingPoseFilter:
                if file.endswith('_raw_filter_data_weaving.csv'):
                    filterData = pd.read_csv(dataPath + file)
            else:
                if file.endswith('_raw_filter_data_lookaroud.csv'):
                    filterData = pd.read_csv(dataPath + file)


        newDataPointTimestamp = filterData['latest_time'][0]
        x = filterData['latest_x'][0]
        y = filterData['latest_y'][0]
        z = filterData['latest_z'][0]
        t = newDataPointTimestamp

        history.append(SR_vector4d(x=x, y=y, z=z, t=t))

        for j in range(1, len(filterData)):
            currentTime = filterData['current_time'][j]
            if (filterData['latest_time'][j] > newDataPointTimestamp):
                newDataPointTimestamp = filterData['latest_time'][j]
                x = filterData['latest_x'][j]
                y = filterData['latest_y'][j]
                z = filterData['latest_z'][j]
                t = newDataPointTimestamp

                if (len(history) >= maxHistorySize):
                    del history[0]

                history.append(SR_vector4d(x=x, y=y, z=z, t=t))

            predictedFacePosition = SR_vector3d(x=0, y=0, z=0)
            if (len(history) >= 3):
                if self.filterParameters.filterType == FilterType.WeavingPoseFilter or self.filterParameters.filterType == FilterType.LookaroundFilter:
                    self.predict1(currentTime=currentTime, history=history, output=output)
                elif self.filterParameters.filterType == FilterType.LastVelosity:
                    self.predict2(currentTime=currentTime, history=history, output=output)
                elif self.filterParameters.filterType == FilterType.PolynomialFit:
                    self.predict3(currentTime=currentTime,history=history, output=output)

        return output.predicted_t, output.predicted_x, output.predicted_y, output.predicted_z


def calculate_prediction_error(t_predicted, signal_predicted, t_ground_truth, signal_ground_truth):
    ground_truth_at_predicted = np.interp(t_predicted, t_ground_truth, signal_ground_truth)
    error = signal_predicted - ground_truth_at_predicted
    return error















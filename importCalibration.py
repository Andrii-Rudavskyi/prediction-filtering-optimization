import yaml
import os
import numpy as np
import yaml.loader
import cv2 as cv

class cameraIntrinsics:
    def __init__(self, fx=330, fy=330, cx=320, cy=240, d=[0,0,0,0,0], width=640, height=480):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.d = d
        self.width = width
        self.height = height

class Extrinsics:
    def __init__(self, r=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], t=[0, 0, 0], f=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        self.r = r
        self.t = t
        self.f = f

class stereoCameraCalibrationData:
    def __init__(self, leftIntrinsics=cameraIntrinsics(), rightIntrinsics=cameraIntrinsics(), extrinsics=Extrinsics()):
        self.left = leftIntrinsics
        self.right = rightIntrinsics
        self.extrinsics = extrinsics


def importCalibration(calibrationDataPath):
    intrinsicsFilePath = calibrationDataPath + '/' + 'intrinsics.yml'
    fs = cv.FileStorage(intrinsicsFilePath, cv.FILE_STORAGE_READ)
    m1 = fs.getNode('M1')
    m2 = fs.getNode('M2')
    d1 =fs.getNode('D1')
    d2 =fs.getNode('D2')

    leftCamera = cameraIntrinsics(fx=m1.mat()[0,0], fy=m1.mat()[1,1], cx=m1.mat()[0,2], cy=m1.mat()[1,2], d=d1.mat()[0], width=640, height=480)
    rightCamera = cameraIntrinsics(fx=m2.mat()[0,0], fy=m2.mat()[1,1], cx=m2.mat()[0,2], cy=m2.mat()[1,2], d=d2.mat()[0], width=640, height=480)
    fs.release()

    extrnsicsFilePath = calibrationDataPath + '/' + 'extrinsics.yml'
    fs = cv.FileStorage(extrnsicsFilePath, cv.FILE_STORAGE_READ)
    r = fs.getNode('R')
    t = fs.getNode('T')
    f = fs.getNode('F')

    extrinsics = Extrinsics(r=r.mat(), t=t.mat(), f=f.mat())

    fs.release()

    stereoCamera = stereoCameraCalibrationData(leftIntrinsics=leftCamera, rightIntrinsics=rightCamera, extrinsics=extrinsics)

    print("------------------------------------------")
    print("Camera calibration data have been imported")
    print("Left camera: ")
    print("fx: ", stereoCamera.left.fx,  " | fy: ", stereoCamera.left.fy, " | cx: ", stereoCamera.left.cx, " | cy: ", stereoCamera.left.cy, " | width: ", stereoCamera.left.width, " | height: ", stereoCamera.left.height)
    print("Distorsion coefficients: ", stereoCamera.left.d)
    print("")
    print("Right camera: ")
    print("fx: ", stereoCamera.right.fx,  " | fy: ", stereoCamera.right.fy, " | cx: ", stereoCamera.right.cx, " | cy: ", stereoCamera.right.cy, " | width: ", stereoCamera.right.width, " | height: ", stereoCamera.right.height)
    print("Distorsion coefficients: ", stereoCamera.right.d)
    print("Extrinsics:")
    print("Rotation matrix: \n", stereoCamera.extrinsics.r)
    print("Translation: \n", stereoCamera.extrinsics.t)
    print("F: \n", stereoCamera.extrinsics.f)
    print("------------------------------------------")

    return stereoCamera
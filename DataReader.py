import os
import cv2
import numpy as np

class DataReader:
    def __init__(self, datasetDirectory):
        self._datasetDirectory = datasetDirectory
        self._imagesCount = len([i for i in os.listdir(datasetDirectory) if i.endswith(".png")])
        print(f"Found {self._imagesCount} images in {self._datasetDirectory} dataset.")

    def readFrame(self, idx=1, convertToRGB=False):
        frame = cv2.imread(os.path.join(self._datasetDirectory, "temple{:04d}.png".format(idx)))
        if frame is None:
            raise Exception("Cannot read frame", idx)
        if convertToRGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def readCameraParams(self, idx=1):
        with open(os.path.join(self._datasetDirectory, "temple_par.txt")) as f:
            params = f.readlines()[idx].rstrip()
            params = list(map(float, params.split(" ")[1:]))
            K = np.zeros((3, 3))
            R = np.zeros((3, 3)) 
            t = np.zeros((1, 3))
            K[0,0], K[0,1], K[0,2], K[1,0], K[1,1], K[1,2], K[2,0], K[2,1], K[2,2] = params[0:9]
            R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2] = params[9:18]
            t[0,0], t[0,1], t[0,2] = params[18:]
            return K, R, t
        
    def getImagesCount(self):
        return self._imagesCount
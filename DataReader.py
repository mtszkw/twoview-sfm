import os
import cv2
import numpy as np

class DataReader:
    def __init__(self, datasetDirectory):
        self._datasetDirectory = datasetDirectory
        self._imageExtension = ".JPG"
        self._calibDirectory = os.path.join(datasetDirectory, "dslr_calibration_undistorted")
        self._imagesDirectory = os.path.join(datasetDirectory, os.path.join("images", "dslr_images_undistorted"))

        self._imagesCount = len([i for i in os.listdir(self._imagesDirectory) if i.endswith(self._imageExtension)])
        print(f"Found {self._imagesCount} images in {self._imagesDirectory} dataset.")

    def readFrame(self, name, convertToRGB=False, scale=1.0):
        frame = cv2.imread(os.path.join(self._imagesDirectory, name) + self._imageExtension)
        if frame is None:
            raise Exception("Cannot read frame", name)
        if convertToRGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        return frame

    # def readCameraParams(self, idx=1):
    #     with open(os.path.join(self._calibDirectory, "temple_par.txt")) as f:
    #         params = f.readlines()[idx].rstrip()
    #         params = list(map(float, params.split(" ")[1:]))
    #         K = np.zeros((3, 3))
    #         R = np.zeros((3, 3)) 
    #         t = np.zeros((1, 3))
    #         K[0,0], K[0,1], K[0,2], K[1,0], K[1,1], K[1,2], K[2,0], K[2,1], K[2,2] = params[0:9]
    #         R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2] = params[9:18]
    #         t[0,0], t[0,1], t[0,2] = params[18:]
    #         return K, R, t
        
    def getImagesCount(self):
        return self._imagesCount
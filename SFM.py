import cv2
import numpy as np

from DataReader import DataReader
from Filtering import LoweRatioFilter, inlierMaskFilter
from PointCloud import savePointCloud, normalizePointsWithCameraMatrix


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    # Source: http://vision.middlebury.edu/mview/data/
    # Structure of datasetDirectory:
    # /data/temple
    # |_ temple_ang.txt
    # |_ temple_par.txt
    # |_ temple0001.png
    # |_ ...
    dataReader = DataReader(datasetDirectory="data/temple/")

    matcher = cv2.BFMatcher()
    sift = cv2.xfeatures2d.SIFT_create()
    K, _, _ = dataReader.readCameraParams()
    cameraPosition = np.array([[0], [0], [0], [1]])
    print(f"\nIntrinsic parameters matrix:\n{K}")    
    
    points3D, colors3D = [], []
    cameras3D, truthCameras3D = [], []

    firstFrame, numFrames = 40, 10

    for frameIdx in range(firstFrame, firstFrame+numFrames):
        prevFrame = dataReader.readFrame(frameIdx, convertToRGB=True)
        currFrame = dataReader.readFrame(frameIdx+1, convertToRGB=True)
        prevKeypts, prevDescr = sift.detectAndCompute(prevFrame, mask=None)
        currKeypts, currDescr = sift.detectAndCompute(currFrame, mask=None)

        matches = matcher.knnMatch(queryDescriptors=prevDescr, trainDescriptors=currDescr, k=2)
        print(f"\nMatching: Found {len(matches)} matches betweens frames ({frameIdx}, {frameIdx+1}).")

        _, prevPts, currPts = LoweRatioFilter(matches, prevKeypts, currKeypts)

        E, inlierMask = cv2.findEssentialMat(currPts, prevPts, K, cv2.RANSAC, 0.99, 1.0, None)
        _, R, t, inlierMask = cv2.recoverPose(E, currPts, prevPts, K, mask=inlierMask)
        print(f"Rotation R = {R.flatten()}\nTranslation t = {t.flatten()}\nEssential matrix: = {E.flatten()}")
            
        # Filter outliers using a mask obtained from RANSAC
        prevPts, currPts = inlierMaskFilter(prevPts, currPts, inlierMask)
        if len(prevPts) == 0:
            continue

        # Computing current camera position
        R1 = np.vstack(( R, np.zeros((1, 3)) ))
        t1 = np.vstack(( t, np.ones((1, 1)) ))
        R1t1 = np.hstack((R1, t1))
        cameraPosition = np.matmul(R1t1, cameraPosition)

        # Normalize points (using camera matrix K) and compute their 3D positions, then convert to heterogeneous coords.
        normPrevPtsT = np.transpose(normalizePointsWithCameraMatrix(prevPts, K))
        normCurrPtsT = np.transpose(normalizePointsWithCameraMatrix(currPts, K))
        reprojectedPts = cv2.triangulatePoints(np.eye(3, 4), np.hstack((R, t)), normPrevPtsT, normCurrPtsT)
        reprojectedPts = [[x/w, y/w, z/w] for [x, y, z, w] in np.transpose(reprojectedPts)]

        # Save ground truth camera position (translation) for this frame        
        truthCameras3D.append(dataReader.readCameraParams(frameIdx)[2].flatten())

        # Save 3D point and its computed color for this frame
        points3D += reprojectedPts
        cameras3D.append(cameraPosition.flatten())
        colors3D += [prevFrame[int(pt[1]), int(pt[0])] for pt in prevPts]

    savePointCloud(points3D, colors3D, "sfm_cloud.ply")
    savePointCloud(cameras3D, [(255,255,255)]*len(cameras3D), "sfm_cameras.ply")
    savePointCloud(truthCameras3D, [(50,255,50)]*len(truthCameras3D), "sfm_truthCameras.ply")

    print(f"\nReconstructed total {len(points3D)} points from {numFrames} frames.")  

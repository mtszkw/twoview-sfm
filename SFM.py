import cv2
import plyfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataReader import DataReader

def filterWithLoweRatio(matches, queryKeypts, trainKeypts, ratio=0.8):
    goodMatches = [m for m, n in matches if m.distance < 0.8*n.distance]        
    queryPts = np.array([queryKeypts[m.queryIdx].pt for m in goodMatches])
    trainPts = np.array([trainKeypts[m.trainIdx].pt for m in goodMatches])
    print(f"Lowe-Ratio filter: {len(goodMatches)} matches left (ratio = 0.8).")
    return (goodMatches, queryPts, trainPts)


def normalizePointsWithCameraMatrix(points, K):
    focal, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    return np.array([ [(p[0]-cx)/focal, (p[1]-cy)/focal] for p in points])


def savePointCloud(points3D, colors3D, outputFile):
    vertexes = [ (p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(points3D, colors3D)]
    vertexes = [ v for v in vertexes if v[2] >= 0 ] # Discard negative z
    print(f"\nSample 3D points with colors:\n", vertexes[0])
    dtypes = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    array = np.array(vertexes, dtype=dtypes)
    element = plyfile.PlyElement.describe(array, "vertex")
    plyfile.PlyData([element]).write(outputFile)
    

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    dataReader = DataReader(datasetDirectory="data/temple/") # http://vision.middlebury.edu/mview/data/

    matcher = cv2.BFMatcher()
    sift = cv2.xfeatures2d.SIFT_create()
    K, _, _ = dataReader.readCameraParams()
    print(f"\nIntrinsic parameters matrix:\n{K}")

    cloudPoints, cloudColors = [], []
    imagesCount = dataReader.getImagesCount() - 1

    for frameIdx in range(2, imagesCount):
        prevFrame = dataReader.readFrame(frameIdx-1, convertToRGB=True)
        currFrame = dataReader.readFrame(frameIdx, convertToRGB=True)
        prevKeypts, prevDescr = sift.detectAndCompute(prevFrame, mask=None)
        currKeypts, currDescr = sift.detectAndCompute(currFrame, mask=None)

        matches = matcher.knnMatch(queryDescriptors=prevDescr, trainDescriptors=currDescr, k=2)
        print(f"\nMatching: Found {len(matches)} matches betweens frames ({frameIdx-1}, {frameIdx}).")

        _, prevPts, currPts = filterWithLoweRatio(matches, prevKeypts, currKeypts)

        E, inlierMask = cv2.findEssentialMat(prevPts, currPts, K, cv2.RANSAC, 0.99, 1.0, None)
        _, R, t, inlierMask = cv2.recoverPose(E, prevPts, currPts, K, mask=inlierMask)
        prevPts = np.array([pt for (idx, pt) in enumerate(prevPts) if inlierMask[idx] == 1])
        currPts = np.array([pt for (idx, pt) in enumerate(currPts) if inlierMask[idx] == 1])
        print(f"RANSAC filter: {len(prevPts)} keypoints left after applying inlier mask.")
        print(f"Rotation R = {R.flatten()}\nTranslation t = {t.flatten()}\nEssential matrix: = {E.flatten()}")

        if len(prevPts) == 0:
            continue

        normPrevPtsT = np.transpose(normalizePointsWithCameraMatrix(prevPts, K))
        normCurrPtsT = np.transpose(normalizePointsWithCameraMatrix(currPts, K))
        points3D = cv2.triangulatePoints(np.eye(3, 4), np.hstack((R, t)), normPrevPtsT, normCurrPtsT)
        points3D = [[x/w, y/w, z/w] for [x, y, z, w] in np.transpose(points3D)] # Convert to heterogeneous
        
        cloudPoints += points3D
        cloudColors += [prevFrame[int(pt[1]), int(pt[0])] for pt in prevPts]

    savePointCloud(cloudPoints, cloudColors, "sfm_cloud.ply")
    print(f"\nReconstructed total {len(cloudPoints)} points from {imagesCount} frames.")  

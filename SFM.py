import cv2
import plyfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataReader import DataReader

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

# http://vision.middlebury.edu/mview/data/
dataReader = DataReader(datasetDirectory="data/temple/")

matcher = cv2.BFMatcher()
extractor = cv2.xfeatures2d.SIFT_create()
K, _, _ = dataReader.readCameraParams(idx=1)
print(f"\nIntrinsic parameters matrix:\n{K}")

cloudPoints, cloudColors = [], []
imagesCount = 30 # dataReader.getImagesCount() - 1

for frameIdx1 in range(1, imagesCount):

    prevFrame = dataReader.readFrame(frameIdx1, convertToRGB=True)
    currKeypts, prevDescr = extractor.detectAndCompute(prevFrame, mask=None)

    for frameIdx2 in range(frameIdx1+1, imagesCount):
        
        currFrame = dataReader.readFrame(frameIdx2, convertToRGB=True)
        prevKeypts, currDescr = extractor.detectAndCompute(currFrame, mask=None)

        matches = matcher.knnMatch(queryDescriptors=prevDescr, trainDescriptors=currDescr, k=2)
        print(f"\nMatching: Found {len(matches)} matches betweens frames ({frameIdx1}, {frameIdx2}).")

        goodMatches = [m for m, n in matches if m.distance < 0.8*n.distance]        
        prevPts = [currKeypts[m.queryIdx].pt for m in goodMatches]
        currPts = [prevKeypts[m.trainIdx].pt for m in goodMatches]
        print(f"Lowe-Ratio filter: {len(goodMatches)} matches left (ratio = 0.8).")

        # Filtering outliers using fundamental matrix
        _, inlierMask = cv2.findFundamentalMat(np.array(prevPts), np.array(currPts), cv2.FM_RANSAC, 1, 0.99)
        prevPts = np.array([pt for (idx, pt) in enumerate(prevPts) if inlierMask[idx] == 1])
        currPts = np.array([pt for (idx, pt) in enumerate(currPts) if inlierMask[idx] == 1])
        print(f"RANSAC filter: {len(prevPts)} keypoints left after applying inlier mask.")

        E, mask = cv2.findEssentialMat(currPts, prevPts, K, cv2.RANSAC, 0.99, 1.0, None)
        _, R, t, mask = cv2.recoverPose(E, currPts, prevPts, K)
        print("Rotation R =\n", R)
        print("Translation t =", t.flatten())
        print("Essential matrix:\n", E)

        P1 = np.eye(3,4)
        P2 = np.hstack((R, t))
        points3D = cv2.triangulatePoints(P1, P2, np.transpose(prevPts), np.transpose(currPts))
        points3D = [[x/w, y/w, z/w] for [x, y, z, w] in np.transpose(points3D)] # Convert to heterogeneous
        
        cloudPoints += points3D
        cloudColors += [prevFrame[int(pt[1]), int(pt[0])] for pt in prevPts]

print(f"\nReconstructed total {len(cloudPoints)} points from {imagesCount} frames.")

vertexes = [ (p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(cloudPoints, cloudColors)]
dtypes = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
array = np.array(vertexes, dtype=dtypes)
element = plyfile.PlyElement.describe(array, "vertex")
plyfile.PlyData([element]).write("sfm_cloud.ply")

# print(f"\nSample 3D points with colors:\n", np.hstack((cloudPoints, cloudColors))[:10])
# PyntCloud(pd.DataFrame(data=np.hstack((cloudPoints, cloudColors)), columns=["x", "y", "z", "red", "green", "blue"])).to_file("sfm_cloud.ply")

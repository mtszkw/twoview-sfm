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

    goodMatches = [m for m, n in matches if m.distance < 0.8*n.distance]        
    prevPts = [prevKeypts[m.queryIdx].pt for m in goodMatches]
    currPts = [currKeypts[m.trainIdx].pt for m in goodMatches]
    print(f"Lowe-Ratio filter: {len(goodMatches)} matches left (ratio = 0.8).")

    prevPts = np.array(prevPts)
    currPts = np.array(currPts)
    E, inlierMask = cv2.findEssentialMat(currPts, prevPts, K, cv2.RANSAC, 0.99, 1.0, None)
    _, R, t, inlierMask = cv2.recoverPose(E, currPts, prevPts, K, mask=inlierMask)
    prevPts = np.array([pt for (idx, pt) in enumerate(prevPts) if inlierMask[idx] == 1])
    currPts = np.array([pt for (idx, pt) in enumerate(currPts) if inlierMask[idx] == 1])
    print(f"RANSAC filter: {len(prevPts)} keypoints left after applying inlier mask.")
    print("Rotation R =", R.flatten())
    print("Translation t =", t.flatten())
    print("Essential matrix:", E.flatten())

    if len(prevPts) == 0:
        continue

    P1 = np.eye(3, 4)
    P2 = np.hstack((R, t))
    focal, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    normPrevPts = np.array([ [(p[0]-cx)/focal, (p[1]-cy)/focal] for p in prevPts])
    normCurrPts = np.array([ [(p[0]-cx)/focal, (p[1]-cy)/focal] for p in currPts])
    points3D = cv2.triangulatePoints(P1, P2, np.transpose(normPrevPts), np.transpose(normCurrPts))
    points3D = [[x/w, y/w, z/w] for [x, y, z, w] in np.transpose(points3D)] # Convert to heterogeneous
    
    cloudPoints += points3D
    cloudColors += [prevFrame[int(pt[1]), int(pt[0])] for pt in prevPts]

print(f"\nReconstructed total {len(cloudPoints)} points from {imagesCount} frames.")

vertexes = [ (p[0], p[1], p[2], c[0], c[1], c[2]) for p, c in zip(cloudPoints, cloudColors)]
vertexes = [ v for v in vertexes if v[2] >= 0 ] # Discard negative z
dtypes = [('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
array = np.array(vertexes, dtype=dtypes)
element = plyfile.PlyElement.describe(array, "vertex")
plyfile.PlyData([element]).write("sfm_cloud.ply")

print(f"\nSample 3D points with colors:\n", vertexes[:10])

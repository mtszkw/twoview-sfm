import numpy as np

def LoweRatioFilter(matches, queryKeypts, trainKeypts, ratio=0.8):
    goodMatches = [m for m, n in matches if m.distance < 0.8*n.distance]        
    queryPts = np.array([queryKeypts[m.queryIdx].pt for m in goodMatches])
    trainPts = np.array([trainKeypts[m.trainIdx].pt for m in goodMatches])
    print(f"Lowe-Ratio filter: {len(goodMatches)} matches left (ratio = 0.8).")
    return (goodMatches, queryPts, trainPts)


def inlierMaskFilter(prevPts, currPts, inlierMask):
    prevPts = np.array([pt for (idx, pt) in enumerate(prevPts) if inlierMask[idx] == 1])
    currPts = np.array([pt for (idx, pt) in enumerate(currPts) if inlierMask[idx] == 1])
    print(f"RANSAC filter: {len(prevPts)} keypoints left after applying inlier mask.")
    return prevPts, currPts

import cv2
import numpy as np

def generateDisparityMap(frameGrayL, frameGrayR):
    left_matcher = cv2.StereoSGBM_create(numDisparities=16, blockSize=3)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.2)

    dispL = left_matcher.compute(frameGrayL, frameGrayR)
    dispR = right_matcher.compute(frameGrayR, frameGrayL)
    dispL = np.int16(dispL)
    dispR = np.int16(dispR)
    disparityMap = wls_filter.filter(dispL, frameGrayL, None, dispR)

    disparityMap = cv2.normalize(src=disparityMap, dst=disparityMap, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    disparityMap = np.uint8(disparityMap)
    return disparityMap
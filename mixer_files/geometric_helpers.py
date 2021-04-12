import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
'''
Helper function to transform 3D affine point clouds
'''


def bounding_box_transform(set1, set2):
    src = np.array([set1], copy=True).astype(np.float32)
    dst = np.array([set2], copy=True).astype(np.float32)

    t = cv2.estimateAffine3D(src, dst, confidence=0.99)

    return t[1]





import cv2
import numpy as np 

bgr = cv2.imread('obs.jpg')

def white_mask(bgr):
    wi = np.mean(bgr, axis=2) > 60 
    _var = np.var(bgr, axis=2) < 60
    wi = np.logical_and(wi, _var)
    wi = wi.astype(np.uint8) * 255
    return wi

def yellow_mask(bgr):
    wi = np.mean(bgr, axis=2) > 60 
    _var = np.var(bgr, axis=2) > 60
    wi = np.logical_and(wi, _var)
    wi = wi.astype(np.uint8) * 255
    return wi

wi = white_mask(bgr)
cv2.imshow('wi',wi)
cv2.waitKey(0)

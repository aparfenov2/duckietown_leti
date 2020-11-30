# Solution: Parfenov Aleksey

# from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2
import logging

import time
import warnings
import random


def apply_filters(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask_bottom = np.zeros_like(image)
    mask_bottom[image.shape[0]//2:,:] = 1
    image = cv2.bitwise_and(image, image, mask = mask_bottom)
    _, mask = cv2.threshold(image,60,255,cv2.THRESH_BINARY_INV)
    # lower, upper = [0, 0, 0], [255, 80, 80]
    # lower = np.array(lower, dtype = "uint8")
    # upper = np.array(upper, dtype = "uint8")
    # mask = cv2.inRange(image_hsv, lower, upper)
    # mask_bottom = np.zeros_like(mask)
    # mask_bottom[mask.shape[0]//2:,:] = 1
    # mask = mask & mask_bottom
    # mask = ~mask & mask_bottom
    # cv2.imshow('mask', mask)
    # output = cv2.bitwise_and(image, image, mask = mask)
    # Remove noise
    kernel_erode = np.ones((4,4), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
    kernel_dilate = np.ones((6,6),np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)
    return dilated_mask

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    # print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b


def detect_offset_from_centerline(image):

    def white_mask(bgr):
        wi = np.mean(bgr, axis=2) > 60 
        _var = np.var(bgr, axis=2) < 60
        wi = np.logical_and(wi, _var)
        # wi = wi.astype(np.uint8) * 255
        return wi

    def yellow_mask(bgr):
        wi = np.mean(bgr, axis=2) > 60 
        _var = np.var(bgr, axis=2) > 60
        wi = np.logical_and(wi, _var)
        # wi = wi.astype(np.uint8) * 255
        return wi

    res = cv2.resize(image, (160, 120//2))        
    ymsk = yellow_mask(res)
    wmsk = white_mask(res)
    hmap = np.zeros(res.shape[:2], dtype=np.uint8)
    hmap[ymsk] = 60
    hmap[wmsk] = 255
    hmap2 = hmap[hmap.shape[0]//2:,:]
    # cv2.imshow('hmap', hmap)

    hmap_has_yellow = np.any(hmap2 == 60)

    yc  = 2
    row = hmap2[yc,:]
    yex0, yex1 = -1, -1
    wx0, wx1 = -1, -1
    nzye = (row == 60).nonzero()[0]
    if len(nzye) > 0:
        yex0, yex1 = nzye[0], nzye[-1]
    nzw = (row == 255).nonzero()[0]
    if len(nzw) > 0:
        wx0, wx1 = nzw[0], nzw[-1]
    
    xc = hmap.shape[1]//2
    # yellow & white
    if yex0 >= 0 and wx0 >=0:
        if wx1 > yex1:
            print('YW_C')
            xc = (yex1 + wx1)//2
        else:
            print('YW_Y')
            xc = yex1
    # only yellow
    elif yex0 >= 0 and wx0 < 0:
        print('Y')
        xc = yex1
    # only white
    elif yex0 < 0 and wx0 >= 0:
        if hmap_has_yellow:
            # print('W_Y0')
            for yc in range(yc+1, hmap2.shape[0]//2):
                row = hmap2[yc,:]
                nzye = (row == 60).nonzero()[0]
                if len(nzye) > 0:
                    yex0, yex1 = nzye[0], nzye[-1]
                    xc = yex1
                    print('W_Y')
                    break
        else:
            print('W')
            xc = wx0 - hmap.shape[1]//6

    xc = np.clip(xc, 0, hmap2.shape[1]-1)
    yc = np.clip(yc, 0, hmap2.shape[0]-1)
    hmap2[yc-1:yc+1,xc-1:xc+1] = 127
    cv2.imshow('hmap2', hmap2)

    yc += hmap2.shape[0]
    xc = int(xc * image.shape[1] / hmap.shape[1])
    yc = int(yc * image.shape[0] / hmap.shape[0])
    xc = np.clip(xc, 0, image.shape[1]-1)
    yc = np.clip(yc, 0, image.shape[0]-1)
    return xc, yc

def angle(vector1, vector2):
    """ Returns the angle in radians between given vectors"""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(
        np.stack((v1_u[-2:], v2_u[-2:]))
    )
    if minor == 0:
        sign = 1
    else:
        sign = -np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    return sign * np.arccos(dot_p)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def lpf(hist, _len, v):
    if len(hist) >= _len:
        hist.pop(0)
    hist += [v]
    return np.median(hist)


class DontCrushDuckieTaskSolution:

    def __init__(self, env):
        self.env = env

    def solve(self):
        env = self.env

        condition = True
        action = [0, 0]
        # pid = PID(1, 0.1, 0.05)
        # pid = PID(3, 0.5, 0)
        # pidv = PID(1, 0, 0)
        fail_cnt = 0
        lpf_a = []

        last_blob_x = 0
        while condition:
            obs, reward, done, info = env.step(action)
            # if random.uniform(0,1) > 0.3:
            #     continue
            if done:
                obs = env.reset()
            # env.render()
            image = np.ascontiguousarray(obs)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow('image', image)

            # yellow = apply_filters(image)
            # cv2.imshow('yellow',yellow)
            ret = detect_offset_from_centerline(image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            # continue
            signed_angle = 0
            if ret is not None:
                x, y = ret
                last_blob_x = x
                fail_cnt = 0
                cv2.circle(image, (x,y), radius=3, color=(0, 0, 255), thickness=2)

                a = np.array([x, y], dtype=float)
                c = np.array([image.shape[1]/2, 0], dtype=float)
                b = np.array([image.shape[1]/2, image.shape[0]], dtype=float)
                ba = a - b
                bc = c - b
                signed_angle = angle(ba, bc)
                # a_ctl = pid(signed_angle)
                # signed_angle = lpf(lpf_a, 3, signed_angle)

                # control = np.sign(ret) * (1-abs(ret))

                a_ctl = -3*signed_angle*abs(signed_angle)
                a_ctl = np.clip(a_ctl, -1, 1)
                v = 1 - 1.1*abs(signed_angle)
                v  = np.clip(v, 0.1, 1)
                action = [v, a_ctl]
            else:
                # we are lost
                print('last_blob_x', last_blob_x)
                action = [0, 1 if last_blob_x < yellow.shape[1]/2 else -1]

            # action = [_clamp(action[0],[0,1]), _clamp(action[1],[-1,1])]

            print('angle', signed_angle, 'action', action)

            cv2.imshow('image',image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                env.reset()

from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2

def apply_filters(image_hsv):
    lower, upper = [0, 70, 70], [90, 255, 255]
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    mask = cv2.inRange(image_hsv, lower, upper)
    mask_bottom = np.zeros_like(mask)
    mask_bottom[mask.shape[0]//2:,:] = 1
    mask = mask & mask_bottom
    # output = cv2.bitwise_and(image, image, mask = mask)
    # Remove noise
    kernel_erode = np.ones((4,4), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
    kernel_dilate = np.ones((6,6),np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)
    return dilated_mask

def detect_offset_from_centerline(dilated_mask):
    # Find the different contours
    contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort by area (keep only the biggest one)
    if len(contours) < 2:
        return None
    contours = sorted(contours, key=cv2.contourArea) # smallest first
    contours = contours[:3] # 3 with smallest qrea

    centers = [(int(M['m10']/M['m00']), int(M['m01']/M['m00'])) for cont in contours for M in [cv2.moments(cont)] ]
    cp = np.array([dilated_mask.shape[1]/2, dilated_mask.shape[0]])
    centers = sorted(centers, key=lambda c: np.linalg.norm(np.array(c) - cp), reverse=True) # most distanced first
    return centers[0]

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

class LfChallengeTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task['env']
        # getting the initial picture
        obs, _, _, _ = env.step([0,0])
        # convect in for work with cv
        img = cv2.cvtColor(np.ascontiguousarray(obs), cv2.COLOR_BGR2RGB)
        
        # add here some image processing and calc vel and angle
        action = [0, 0]
        fail_cnt = 0
        last_blob_x = 0

        while True:
            obs, reward, done, info = env.step(action)
            img = cv2.cvtColor(np.ascontiguousarray(obs), cv2.COLOR_BGR2RGB)
            # add here some image processing and calc vel and angle
            image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            yellow = apply_filters(image_hsv)

            ret = detect_offset_from_centerline(yellow)

            signed_angle = 0
            if ret is not None:
                x, y = ret
                last_blob_x = x
                fail_cnt = 0

                a = np.array([x, y], dtype=float)
                c = np.array([img.shape[1]/2, 0], dtype=float)
                b = np.array([img.shape[1]/2, img.shape[0]], dtype=float)
                ba = a - b
                bc = c - b
                signed_angle = angle(ba, bc)

                a_ctl = -3*signed_angle*abs(signed_angle)
                a_ctl = np.clip(a_ctl, -1, 1)
                v = 1 - 1.1*abs(signed_angle)
                v  = np.clip(v, 0.1, 1)
                action = [v, a_ctl]                
            else:
                # we are lost
                print('last_blob_x', last_blob_x)
                action = [0, 1 if last_blob_x < yellow.shape[1]/2 else -1]

            print('angle', signed_angle, 'action', action)

            env.render()
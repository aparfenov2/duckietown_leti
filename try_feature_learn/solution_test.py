
import numpy as np
import cv2
import logging
import torch
import torch.nn as nn
import io
import base64

# from weights import b64str
from model1 import Net
import torchvision.transforms as transforms

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

class NetTeacher:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def predict(self, obs):
        # expects RGB
        _obs = np.ascontiguousarray(obs)
        _obs = self.transform(obs).unsqueeze(0)
        hmap = self.model(_obs.to(self.device)).squeeze().detach().cpu().numpy()
        cv2.imshow('hmap', hmap)
        cv2.waitKey(1)        
        # _max = np.unravel_index(np.argmax(hmap), hmap.shape)
        thd = 0.2
        xc = 0 # turn left if lost, hmap.shape[1]//2
        yc = hmap.shape[0]*3//4
        while yc < hmap.shape[0]:
            _row = hmap[yc,:]
            _nonzero = (_row > thd).nonzero()[0]
            if len(_nonzero) > 0:
                dist_from_left = _nonzero[0]
                dist_from_right = hmap.shape[1] - _nonzero[-1]
                if dist_from_left < dist_from_right:
                    xc = dist_from_left
                else:
                    xc = _nonzero[-1]
                break
            yc += 1        
        xc = int(xc * obs.shape[1] / hmap.shape[1])
        yc = int(yc * obs.shape[0] / hmap.shape[0])

        a = np.array([xc, yc], dtype=float)
        c = np.array([obs.shape[1]//8, 0], dtype=float)
        b = np.array([obs.shape[1]//8, obs.shape[0]], dtype=float)
        ba = a - b
        bc = c - b
        signed_angle = angle(ba, bc)

        a_ctl = -3*signed_angle*abs(signed_angle)
        a_ctl = np.clip(a_ctl, -1, 1)

        def v_f(w,sigma):
            return np.exp(-np.power(w,2)/(2*np.power(sigma,2)))

        v = 0.4 * v_f(signed_angle, 0.5)
        v  = np.clip(v, 0.1, 1)
        action = [v, a_ctl]
        print(action)
        # else:
        #     # we are lost
        #     print('last_blob_x', self.last_blob_x)
        #     action = [0, 1 if self.last_blob_x < yellow.shape[1]/2 else -1]
        return action


class DontCrushDuckieTaskSolution:

    def __init__(self, env):
        self.env = env

    def solve(self):
        env = self.env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model = Net()

        # print('decoding {} bytes'.format(len(b64str)))
        # b = io.BytesIO(base64.b64decode(b64str))
        # state_dict = torch.load(b, map_location=self.device)
        state_dict = torch.load('model.pt', map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)

        self.model.eval()
        self.teacher = NetTeacher(self.model, self.device)

        def preprocess_obs(obs):
            return obs

        obs = env.reset()
        obs = preprocess_obs(obs)

        cnt = 0
        while True:
            action = self.teacher.predict(obs)
            obs, reward, done, info = env.step(action)
            obs = preprocess_obs(obs)
            env.render()
            if done: # or cnt % 100 == 0:
                obs = env.reset()
                obs = preprocess_obs(obs)
            cnt += 1
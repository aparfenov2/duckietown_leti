import sys, os
import argparse
import pyglet
import gym
import numpy as np
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from PIL import Image
import torch
import torch.nn as nn
import logging
# from utils.env import launch_env
# from utils.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper
import cv2

def preprocess_obs(obs):
    # expected RGB
    obs = np.ascontiguousarray(obs)
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    _, obs = cv2.threshold(obs,80, 255, cv2.THRESH_BINARY)
    obs = cv2.resize(obs, (160, 120))
    obs = obs[:,:,np.newaxis]
    obs_lo, obs_hi = 0, 255
    obs = (obs - obs_lo) / (obs_hi - obs_lo)
    obs = obs.transpose(2, 0, 1)
    return obs

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
    contours = contours[:4] # 3 with smallest qrea

    centers = [(int(M['m10']/M['m00']), int(M['m01']/M['m00'])) for cont in contours for M in [cv2.moments(cont)] ]
    cp = np.array([dilated_mask.shape[1]/2, dilated_mask.shape[0]])
    centers = sorted(centers, key=lambda c: np.linalg.norm(np.array(c) - cp), reverse=True) # most distanced first
    return centers[-1]

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

class YellowTeacher:
    def __init__(self):
        self.last_blob_x = 0

    def predict(self, obs):
        image_hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
        self.yellow = yellow = apply_filters(image_hsv)
        ret = detect_offset_from_centerline(yellow)

        signed_angle = 0
        if ret is not None:
            x, y = ret
            self.last_blob_x = x
            fail_cnt = 0

            a = np.array([x, y], dtype=float)
            c = np.array([obs.shape[1]/2, 0], dtype=float)
            b = np.array([obs.shape[1]/2, obs.shape[0]], dtype=float)
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
            print('last_blob_x', self.last_blob_x)
            action = [0, 1 if self.last_blob_x < yellow.shape[1]/2 else -1]
        return action

class Main:
    def __init__(self):
        self.mini_batch = []

    def train(self, obs, expert_action):
        obs = preprocess_obs(obs)

        if len(self.mini_batch) > 2048:
            self.mini_batch.pop(0)
        self.mini_batch.append((obs, expert_action))

        if len(self.mini_batch) >= 128:
# train 1 batch
            self.model.train()

            self.optimizer.zero_grad()
            indexes = np.random.choice(list(range(len(self.mini_batch))), 128)
            observations = np.array([self.mini_batch[i][0] for i in indexes])
            actions = np.array([self.mini_batch[i][1] for i in indexes])

            obs_batch = torch.from_numpy(observations).float().to(self.device)
            act_batch = torch.from_numpy(actions).float().to(self.device)
            
            model_actions = self.model(obs_batch)
            loss = (model_actions - act_batch).norm(2).mean()
            loss.backward()
            self.optimizer.step()
    
            loss = loss.item()
            print(f"batch_size {len(self.mini_batch)}, train loss {loss}")


    def main(self):
        from model import Model

        parser = argparse.ArgumentParser()
        parser.add_argument("--env-name", default=None)
        parser.add_argument("--map-name", default="udem1")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--weights")
        args = parser.parse_args()
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
        self.model = Model(action_dim=2, max_action=1.0)


        if args.weights is not None:
            try:
                state_dict = torch.load(args.weights, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception:
                logging.exception('')
                print("failed to load model")

        # weight_decay is L2 regularization, helps avoid overfitting
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0004, weight_decay=1e-3)

        self.model.eval().to(self.device)

        if args.env_name is None:
            env = DuckietownEnv(map_name=args.map_name, domain_rand=False, draw_bbox=False)
        else:
            env = gym.make(args.env_name)

        self.env = env

        self.last_obs = env.reset()
        env.render()

        teacher = YellowTeacher()

        cnt = 0
        while True:
            action = teacher.predict(self.last_obs)
            self.train(self.last_obs, [0.1, action[1]])
            if cnt % 1000 == 0:
                torch.save(self.model.state_dict(), self.args.weights)
                print('weights saved')
            obs, reward, done, info = env.step([0.1, action[1]])
            self.last_obs = obs
            if done:
                self.last_obs = env.reset()
            env.render()
            cnt+=1

if __name__ == '__main__':
    Main().main()

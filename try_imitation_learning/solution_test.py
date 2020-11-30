
import numpy as np
import cv2
import logging
import torch
import torch.nn as nn
import io
import base64

from weights import b64str
from model import Model
from utils.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper


class DontCrushDuckieTaskSolution:

    def __init__(self, env):
        self.env = env        

    def solve(self):
        env = self.env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model = Model(action_dim=2, max_action=1.0)
        self.model.eval()
        # self.model = torch.quantization.quantize_dynamic(
        #     self.model, {nn.Linear}, dtype=torch.qint8
        # )

        print('decoding {} bytes'.format(len(b64str)))
        b = io.BytesIO(base64.b64decode(b64str))
        state_dict = torch.load(b, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)

        # env = ResizeWrapper(env)
        # env = NormalizeWrapper(env)
        # env = ImgWrapper(env)
        # env = ActionWrapper(env)
        # env = DtRewardWrapper(env)

        def preprocess_obs(obs):
            obs = cv2.resize(obs, (160, 120))
            obs_lo, obs_hi = 0, 255
            obs = (obs - obs_lo) / (obs_hi - obs_lo)
            obs = obs.transpose(2, 0, 1)
            return obs

        obs = env.reset()
        obs = preprocess_obs(obs)

        while True:

            obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
            action = self.model(obs)
            action = action.squeeze().data.cpu().numpy()
            action[0] /= 2
            obs, reward, done, info = env.step(action)
            obs = preprocess_obs(obs)
            env.render()
            if done:
                obs = env.reset()
                obs = preprocess_obs(obs)

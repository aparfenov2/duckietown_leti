from gym_duckietown.envs import DuckietownEnv
from learn_by_trainer import YellowTeacher
import cv2
import numpy as np
import os
from PIL import Image
import torch
from model1 import Net
import torchvision.transforms as transforms

transform = transforms.Compose([
        # SquarePad(image_size),
        # transforms.CenterCrop(image_size),
        # transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
state_dict = torch.load('model.pt', map_location=device)
model.load_state_dict(state_dict)
model.eval()

env = DuckietownEnv(map_name='zigzag_dists', domain_rand=False, draw_bbox=False)

obs = env.reset()
env.render()

teacher = YellowTeacher()
while True:

    action = teacher.predict(obs)
    action[0] /= 1.5
    obs, reward, done, info = env.step(action)
    _obs = np.ascontiguousarray(obs)
    # print(_obs.shape) # 480x640
    _obs = transform(obs).unsqueeze(0)
    # print('_obs.shape', _obs.shape)
    hmap = model(_obs).squeeze().detach().cpu().numpy()
    # print('hmap.shape',hmap.shape) # 12,17
    # print(hmap.shape)
    # _max = np.unravel_index(np.argmax(hmap), hmap.shape)
    # yc, xc = _max
    thd = 0.2
    xc = hmap.shape[1]//2
    yc = hmap.shape[0]//2
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
    cv2.circle(obs, (xc, yc), 2, (255,0,0), 2)
    cv2.imshow('obs', obs)
    cv2.imshow('hmap', hmap)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break
    if key == ord('r'):
        obs = env.reset()

    # if done:
    #     obs = env.reset()
    env.render()


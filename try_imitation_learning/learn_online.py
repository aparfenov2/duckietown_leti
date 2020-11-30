import sys, os, json
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
from torch.utils.data import DataLoader
# from utils.env import launch_env
# from utils.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper
from model import Model
import cv2
import unittest

BATCH_SIZE = 128


def preprocess_obs(obs, resize=True):
    obs = np.ascontiguousarray(obs)
    if resize:
        obs = cv2.resize(obs, (160, 120))
    obs_lo, obs_hi = 0, 255
    obs = (obs - obs_lo) / (obs_hi - obs_lo)
    obs = obs.transpose(2, 0, 1)
    return obs

def restore_obs(obs):
    obs = obs.transpose(1, 2, 0)
    obs_lo, obs_hi = 0, 255
    obs = obs*(obs_hi - obs_lo) + obs_lo
    obs = obs.astype(np.uint8)
    return obs

class InMemoryDataset:
    def __init__(self, _path = 'train'):
        self._list = []
        self._path = _path
        if os.path.isdir(self._path):
            self._load_list()

    def _load_list(self):
        observations = {}
        for f in os.listdir(self._path):
            if os.path.splitext(f)[1] == '.jpg':
                _bgr = cv2.imread(f"{self._path}/{f}")
                _rgb = cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB)
                observations[int(os.path.splitext(f)[0])] = preprocess_obs(_rgb, resize=False)

        with open(f"{self._path}/actions.json",'r') as f:
            actions = json.load(f)
        self._list = [(observations[f], np.array(actions[str(f)])) for f in sorted(observations.keys())]
        print(f'dataset restored from {self._path}')

    def _dump_list(self):
        os.makedirs(self._path, exist_ok=True)
        for i, (obs,act) in enumerate(self._list):
            obs = restore_obs(obs)
            _bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{self._path}/{i:04d}.jpg", _bgr)
        actions = { i : list(act) for i, (obs,act) in enumerate(self._list)}
        with open(f"{self._path}/actions.json",'w') as f:
            json.dump(actions, f, indent=4)
        print(f'dataset stored in {self._path}')

    def __getitem__(self, idx):
        return torch.from_numpy(self._list[idx][0]).float(),\
            torch.from_numpy(self._list[idx][1]).float(),

    def __len__(self):
        return len(self._list)

    def add_pair(self, obs, act):
        if len(self._list) > 256 * BATCH_SIZE:
            self._list.pop(0)
        self._list.append((obs, act))

class DatasetUT(unittest.TestCase):
    def test1(self):
        ds = InMemoryDataset()
        obs = np.zeros((480,640,3))
        obs = preprocess_obs(obs)
        ds.add_pair(obs, np.array([0.0,0.0]))
        ds._dump_list()
        ds._load_list()

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.dataset = InMemoryDataset()
        self.dataset_updated = False
        self.train_iter = None
        self.best_loss = 0
        self.last_loss = 0
        self.epochs_to_try = 0
        self.loop_cnt = 0

    def add_pair(self, obs, act):
        self.dataset.add_pair(obs, act)
        self.dataset_updated = True

        if len(self.dataset) > 1:
            self.train_iter = iter(self.train_step(batch_size = BATCH_SIZE))
            self.epochs_to_try = 0

    def train_step(self, batch_size = 64):
        dataloader = DataLoader(self.dataset, shuffle=True, batch_size = batch_size)

        self.model.train()
        self.optimizer.zero_grad()
        losses = []
        for i, (observations, actions) in enumerate(dataloader):            
            obs_batch = observations.float().to(self.device)
            act_batch = actions.float().to(self.device)
        
            model_actions = self.model(obs_batch)
            loss = (model_actions - act_batch).norm(2).mean()
            loss.backward()
            self.optimizer.step()

            loss = loss.item()
            losses.append(loss)
            self.last_loss = np.mean(losses)
            # print(f"len(dset) {len(self.dataset)}, train loss {loss}")
            yield loss

    def loop_entry(self):
        self.loop_cnt += 1
        if self.loop_cnt % 1000 == 0:
            if self.dataset_updated:
                self.dataset._dump_list()
                self.dataset_updated = False

        if self.train_iter is not None:
            try:
                next(self.train_iter)                
            except StopIteration:
                self.train_iter = None
                if self.last_loss > self.best_loss:
                    if self.epochs_to_try < 10:
                        self.train_iter = iter(self.train_step(batch_size = BATCH_SIZE))
                        self.epochs_to_try += 1
                    else:
                        self.best_loss = self.last_loss
                else:
                    self.epochs_to_try = 0
                    self.best_loss = self.last_loss


class Main:
    def __init__(self):
        self.env = None
        self.key_handler = None

    def update(self, dt):
        """
        This function is called at every frame to handle
        movement/stepping and redrawing
        """
        wheel_distance = 0.102
        min_rad = 0.08

        key_handler = self.key_handler
        env = self.env
        expert_action = np.array([0.0, 0.0])
        # action = self.last_action.copy()
        # action[1] = 0.0
        if key_handler[key.UP]:
            expert_action += np.array([0.44, 0])
        if key_handler[key.DOWN]:
            expert_action -= np.array([0.44, 0])
        if key_handler[key.LEFT]:
            expert_action += np.array([0.1, 1])
        if key_handler[key.RIGHT]:
            expert_action -= np.array([-0.1, 1])
        if key_handler[key.SPACE]:
            expert_action = np.array([0, 0])

        # Speed boost
        if key_handler[key.LSHIFT]:
            expert_action *= 1.5

        if key_handler[key.LALT]:
            print('skip train')
            action = expert_action

# if user action
        elif np.any(expert_action != 0.0):
            self.trainer.add_pair(self.last_obs, expert_action)
            action = expert_action
        else:
# predict
            self.model.eval()
            obs = torch.from_numpy(self.last_obs).float().to(self.device).unsqueeze(0)
            action = self.model(obs)
            action = action.squeeze().data.cpu().numpy()
            action[0] = max(0.05, action[0])

        self.trainer.loop_entry()
        print(f"v={action[0]:02.2f} a={action[1]:02.2f} last_loss={self.trainer.last_loss:02.4f} best_loss={self.trainer.best_loss:02.4f}")

        obs, reward, done, info = env.step(action)
        self.last_obs = preprocess_obs(obs)

        # print("step_count = %s, reward=%.3f" % (self.env.unwrapped.step_count, reward))
        # print(f"last_action={self.last_action}")

        if key_handler[key.RETURN]:
            # im = Image.fromarray(obs)
            # im.save("screen.png")
            torch.save(self.model.state_dict(), self.args.weights)
            print('weights saved')


        # if done:
        #     print("done!")
        #     self.last_obs = preprocess_obs(env.reset())
        #     env.render()

        env.render()



    def main(self):

        parser = argparse.ArgumentParser()
        parser.add_argument("--env-name", default=None)
        parser.add_argument("--map-name", default="udem1")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--weights")
        args = parser.parse_args()
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
        self.model = Model(action_dim=2, max_action=1.0)
        # self.model = torch.quantization.quantize_dynamic(
        #     self.model, {nn.Linear}, dtype=torch.qint8
        # )

        if args.weights is not None:
            try:
                state_dict = torch.load(args.weights, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception:
                logging.exception('')
                print("failed to load model")

        # weight_decay is L2 regularization, helps avoid overfitting
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0004, weight_decay=1e-3)

        self.model.eval().to(self.device)
        self.trainer = Trainer(self.model, optimizer, self.device)

        if args.env_name is None:
            env = DuckietownEnv(map_name=args.map_name, domain_rand=False, draw_bbox=False)
        else:
            env = gym.make(args.env_name)

        # env = launch_env()
        # env = ResizeWrapper(env)
        # env = NormalizeWrapper(env)
        # env = ImgWrapper(env)
        # env = ActionWrapper(env)
        # env = DtRewardWrapper(env)

        self.env = env

        self.last_obs = preprocess_obs(env.reset())
        env.render()

        @env.unwrapped.window.event
        def on_key_press(symbol, modifiers):
            """
            This handler processes keyboard commands that
            control the simulation
            """

            if symbol == key.BACKSPACE or symbol == key.SLASH:
                print("RESET")
                self.last_obs = preprocess_obs(env.reset())
                env.render()
            elif symbol == key.PAGEUP:
                env.unwrapped.cam_angle[0] = 0
            elif symbol == key.R:
                self.last_obs = preprocess_obs(env.reset())
            elif symbol == key.Q:
                env.close()
                sys.exit(0)

        # Register a keyboard handler
        self.key_handler = key_handler = key.KeyStateHandler()
        env.unwrapped.window.push_handlers(key_handler)

        pyglet.clock.schedule_interval(self.update, 1.0 / env.unwrapped.frame_rate)

        # Enter main event loop
        pyglet.app.run()

        env.close()

        # obs = env.reset()
        # env.render()

        # while True:
        #     print('human_agent_action', human_agent_action)
        #     obs, recompense, fini, info = env.step([0.1, 0])
        #     if fini:
        #         obs = env.reset()
        #     env.render()

if __name__ == '__main__':
    Main().main()

import time
import sys
import argparse
import math
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
from solution_test import DontCrushDuckieTaskSolution

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--map-name", default="straight_road")
    parser.add_argument("--no-pause", action="store_true", help="don't pause on failure")
    args = parser.parse_args()

    if args.env_name is None:
        env = DuckietownEnv(map_name=args.map_name, domain_rand=False, draw_bbox=False)
    else:
        env = gym.make(args.env_name)
    env.start_pose = [[0.72,  0.,      0.87],0]
    obs = env.reset()
    # env.render()

    sol = DontCrushDuckieTaskSolution(env)
    sol.solve()
    # while True:
    #     lane_pose = env.get_lane_pos(env.cur_pos, env.cur_angle)
    #     distance_to_road_center = lane_pose.dist
    #     angle_from_straight_in_rads = lane_pose.angle_rad
    #     print('lane_pose', lane_pose)

if __name__ == '__main__':
    main()

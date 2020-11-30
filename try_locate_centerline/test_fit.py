import sys, os
import argparse
import pyglet
import gym
import numpy as np
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from PIL import Image
import cv2

def apply_filters(image_hsv):
    # lower, upper = [0, 0, 0], [255, 80, 80]
    # lower = np.array(lower, dtype = "uint8")
    # upper = np.array(upper, dtype = "uint8")
    # mask = cv2.inRange(image_hsv, lower, upper)


    # mask_bottom = np.zeros_like(mask)
    # mask_bottom[mask.shape[0]//2:,:] = 1
    # mask = mask & mask_bottom
    # mask = ~mask & mask_bottom
    # output = cv2.bitwise_and(image, image, mask = mask)
    # Remove noise
    ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv2.imshow('mask', mask)
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
    if denum == 0:
        denum = 0.00001
    b = numer / denum
    a = ybar - b * xbar

    # print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

def cv2_plot(image, x_values, y_values):
    # If you have something like this
    # matplotlib.pyplot.plot(x_values, y_values, color='yellow')

    # You can do the same on OpenCV like this
    curve = np.column_stack((x_values.astype(np.int32), y_values.astype(np.int32)))
    cv2.polylines(image, [curve], False, (0,255,255))

    # And if you need to plot more curves just add them as an element to the array of polygonal curves
    # curve1 = np.column_stack((x1.astype(np.int32), y1.astype(np.int32)))
    # curve2 = np.column_stack((x2.astype(np.int32), y2.astype(np.int32)))
    # cv2.polylines(image, [curve1, curve2], False, (0,255,255))

def do_segment(frame):
    height = frame.shape[0]
    width  = frame.shape[1]
    polygons = np.array([
                            [(0, height), (width, height), (width//2, height//3)]
                        ])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, 255)

    # mask1 = np.zeros_like(frame)
    # mask1[height*2//3:,:] = 1
    return mask # & mask1

def detect_offset_from_centerline(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _,mask = cv2.threshold(image,60,255,cv2.THRESH_BINARY)

    mask_bottom = np.zeros_like(mask)
    mask_bottom[image.shape[0]//2:,:] = 255
    # mask_bottom = do_segment(mask)
    mask = mask & mask_bottom
    kernel_erode = np.ones((4,4), np.uint8)
    eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
    kernel_dilate = np.ones((6,6),np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)
    # Find the different contours
    contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('dilated_mask', dilated_mask)
    # Sort by area (keep only the biggest one)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) < 2:
        return None
    centers = [(int(M['m10']/M['m00']), int(M['m01']/M['m00'])) for cont in contours[-2:-1] for M in [cv2.moments(cont)] ]
    return centers[-1]
    # xc, yc = img.shape[1]/2, img.shape[0]/2
    # centers = sorted(centers, key=lambda c: np.linalg.norm(np.array([xc,yc]) - np.array(c)))
    # centers = centers[-2:-1]
    # x = [c[0] for c in centers]
    # y = [c[1] for c in centers]

    # for _x,_y in centers:
    #     cv2.circle(img, (_x,_y), radius=3, color=(0, 0, 255), thickness=2)

    # z = np.polyfit(x, y, 3)
    # print('z', z)
    # f = np.poly1d(z)
    # disp = np.zeros_like(dilated_mask)
    # x_new = np.linspace(xs[0], xs[-1], 50)
    # y_new = f(x_new)
    # cv2_plot(img, x_new, y_new)

    # a,b = best_fit(x, y)
    # print(a, b)
    # return 0

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

class Main:
    def __init__(self):
        self.env = None
        self.key_handler = None

    def update(self, dt):
        """
        This function is called at every image to handle
        movement/stepping and redrawing
        """
        wheel_distance = 0.102
        min_rad = 0.08

        key_handler = self.key_handler
        env = self.env
        action = np.array([0.0, 0.0])
        # action = self.last_action.copy()
        action[1] = 0.0
        if key_handler[key.UP]:
            action += np.array([0.44, 0])
        if key_handler[key.DOWN]:
            action -= np.array([0.44, 0])
        if key_handler[key.LEFT]:
            action += np.array([0, 1])
        if key_handler[key.RIGHT]:
            action -= np.array([0, 1])
        if key_handler[key.SPACE]:
            action = np.array([0, 0])

        # Speed boost
        if key_handler[key.LSHIFT]:
            action *= 1.5
        self.last_action = action.copy()
        obs, reward, done, info = env.step(action)

        image = np.ascontiguousarray(obs)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        from solution_test import detect_offset_from_centerline as detect_offset_from_centerline1
        xc, yc = detect_offset_from_centerline1(bgr)
        cv2.circle(bgr, (xc,yc), radius=3, color=(0, 0, 255), thickness=2)
        cv2.imshow('bgr', bgr)
        # cv2.imwrite('obs.jpg', bgr)
        # exit(0)

        # def white_mask(bgr):
        #     wi = np.mean(bgr, axis=2) > 60 
        #     _var = np.var(bgr, axis=2) < 60
        #     wi = np.logical_and(wi, _var)
        #     # wi = wi.astype(np.uint8) * 255
        #     return wi

        # def yellow_mask(bgr):
        #     wi = np.mean(bgr, axis=2) > 60 
        #     _var = np.var(bgr, axis=2) > 60
        #     wi = np.logical_and(wi, _var)
        #     # wi = wi.astype(np.uint8) * 255
        #     return wi

        # res = cv2.resize(bgr, (160, 120))        
        # ymsk = yellow_mask(res)
        # wmsk = white_mask(res)
        # hmap = np.zeros(res.shape[:2], dtype=np.uint8)
        # hmap[ymsk] = 100
        # hmap[wmsk] = 255
        # hmap = hmap[hmap.shape[0]//2:,:]
        # cv2.imshow('hmap', hmap)
        # mask_bottom = np.zeros_like(ymsk)
        # mask_bottom[mask_bottom.shape[0]//2:,:] = 255


        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # mask_bottom = np.zeros_like(image)
        # mask_bottom[image.shape[0]//2:,:] = 1
        # image = cv2.bitwise_and(image, image, mask = mask_bottom)

        # ret2,th2 = cv2.threshold(image,60,255,cv2.THRESH_BINARY)
        # cv2.imshow('th', th2)
        # ret = detect_offset_from_centerline(image)
        # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # yellow = apply_filters(image_hsv)
        # cv2.imshow('yellow',yellow)

        # if ret is not None:
        #     x,y = ret
        #     cv2.circle(image, (x,y), radius=3, color=(0, 0, 255), thickness=2)

        #     a = np.array([x, y], dtype=float)
        #     c = np.array([image.shape[1]/2, 0], dtype=float)
        #     b = np.array([image.shape[1]/2, image.shape[0]], dtype=float)
        #     ba = a - b
        #     bc = c - b
        #     signed_angle = angle(ba, bc)
        #     # cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        #     # angle = np.arccos(cosine_angle)
        #     print (np.degrees(signed_angle))

        #     # point_vec =  np.array([x, y], dtype=float) - np.array([image.shape[1]/2, image.shape[0]], dtype=float)
        #     # point_vec /= np.linalg.norm(point_vec)
        #     # right_vec = np.array([0, -image.shape[0]], dtype=float)
        #     # right_vec /= np.linalg.norm(right_vec)

        #     # dot = np.dot(right_vec, point_vec)
        #     # omega = -1 * np.arccos(dot)
        #     # print('omega',omega)

        # cv2.imshow('image',image)

        # print(ret)
        # print("step_count = %s, reward=%.3f" % (self.env.unwrapped.step_count, reward))
        # print(f"last_action={self.last_action}")
        cv2.waitKey(1)


        # if done:
        #     print("done!")
        #     env.reset()
        #     env.render()

        env.render()



    def main(self):

        parser = argparse.ArgumentParser()
        parser.add_argument("--env-name", default=None)
        parser.add_argument("--map-name", default="udem1")
        args = parser.parse_args()

        if args.env_name is None:
            env = DuckietownEnv(map_name=args.map_name, domain_rand=False, draw_bbox=False)
        else:
            env = gym.make(args.env_name)
        self.env = env

        env.reset()
        env.render()

        @env.unwrapped.window.event
        def on_key_press(symbol, modifiers):
            """
            This handler processes keyboard commands that
            control the simulation
            """

            if symbol == key.BACKSPACE or symbol == key.SLASH:
                print("RESET")
                env.reset()
                env.render()
            elif symbol == key.PAGEUP:
                env.unwrapped.cam_angle[0] = 0
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

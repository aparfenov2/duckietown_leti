from gym_duckietown.envs import DuckietownEnv
from learn_by_trainer import YellowTeacher
import cv2
import numpy as np
import os

def intersects(x1,y1,x2,y2, bbx):
    bx1,by1,bx2,by2 = bbx
    bx2 += bx1
    by2 += by1
    # test bbx inside
    if x1 <= bx1 and bx2 <= x2:
        if y1 <= by1 and by2 <= y2:
            return True
    # test inside bbx
    if bx1 <= x1 and x2 <= bx2:
        if by1 <= y1 and y2 <= by2:
            return True
    # test corner inside
    if x1 <= bx1 <= x2:
        if y1 <= by1 <= y2:
            return True
    if x1 <= bx2 <= x2:
        if y1 <= by1 <= y2:
            return True
    if x1 <= bx2 <= x2:
        if y1 <= by2 <= y2:
            return True
    if x1 <= bx1 <= x2:
        if y1 <= by2 <= y2:
            return True
    return False

def pad(x1,y1,x2,y2, NEG_SIZE, img):
    if x2-x1 < NEG_SIZE:
        diff = NEG_SIZE - (x2-x1)
        x1 -= diff//2
        x2 += diff//2
    if y2-y1 < NEG_SIZE:
        diff = NEG_SIZE - (y2-y1)
        y1 -= diff//2
        y2 += diff//2
    x1,x2 = np.clip([x1,x2], 0, img.shape[1])
    y1,y2 = np.clip([y1,y2], 0, img.shape[0])
    return x1,y1,x2,y2

env = DuckietownEnv(map_name='zigzag_dists', domain_rand=False, draw_bbox=False)

obs = env.reset()
env.render()

teacher = YellowTeacher()
os.makedirs('out/1',exist_ok=True)
os.makedirs('out/0',exist_ok=True)
cnt = 0
cnt1 = 0
NEG_SIZE = 128

while True:

    action = teacher.predict(obs)
    ymsk = teacher.yellow
    contours, hierarchy = cv2.findContours(ymsk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 2:
        contours = sorted(contours, key=cv2.contourArea) # smallest first
        contours = contours[:4] # 3 with smallest qrea
    bboxes = [cv2.boundingRect(c) for c in contours]

    img = np.ascontiguousarray(obs)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img1 = img.copy()
    _len1 = 0
    for b in bboxes:
        x1,y1,x2,y2 = b
        if x2 < NEG_SIZE //4 or y2 < NEG_SIZE //4:
            continue
        _len1 += 1
        x2 += x1
        y2 += y1
        x1,y1,x2,y2 = pad(x1,y1,x2,y2, NEG_SIZE, img)
        roi = img1[y1:y2,x1:x2]
        cv2.imwrite(f'out/1/{cnt:05}.jpg', roi)
        cnt += 1
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
    # gen negative

    for _ in range(_len1):
        while True:
            x1 = np.random.randint(0, img.shape[1] - NEG_SIZE - 1)
            y1 = np.random.randint(0, img.shape[0] - NEG_SIZE - 1)
            x2 = x1 + NEG_SIZE
            y2 = y1 + NEG_SIZE
            if not any([intersects(x1, y1, x2, y2, bbx) for bbx in bboxes]):
                roi = img1[y1:y2,x1:x2]
                cv2.imwrite(f'out/0/{cnt1:05}.jpg', roi)
                cnt1 += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
                break

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    action[0] /= 1.5
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
    env.render()

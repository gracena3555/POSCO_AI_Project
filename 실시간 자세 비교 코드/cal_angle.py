from numpy import dot
from numpy.linalg import norm
import numpy as np
from math import degrees, acos

import argparse
import logging
import sys
import time
import math
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 모듈 import
import sys
import time
from keras.models import load_model
from util import *
sys.path.append('/usr/local/python')
from openpose import pyopenpose as op

def angle_between_points( p0, p1, p2 ):
  a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
  b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
  c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
  if a * b == 0:
      return -1.0 
  return  math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180 /math.pi

def length_between_points(p0, p1):
    return math.hypot(p1[0]- p0[0], p1[1]-p0[1])

def get_angle_point(human, pos):
    pnts = []

    # if pos == 'left_elbow':[2021-04-18 23:35:05,354] [TfPoseEstimatorRun] [INFO] right arm angle:42.875220
    # elif pos == 'left_ankle':
    #     pos_list = (5,12,14)
    # elif pos == 'right_elbow':
    #     pos_list = (2,3,4)
    # elif pos == 'right_hand':
    #     pos_list = (1,2,4)
    # elif pos == 'right_knee':
    #     pos_list = (9,10,11)
    # elif pos == 'right_ankle':
    #     pos_list = (2,9,11)
    if pos == 'left_arm': # ok
        pos_list = (8,1,7)
    elif pos == 'right_arm': # ok
        pos_list = (8,1,4)
    elif pos == 'left_leg': # ok
        pos_list = (1,8,14)
    elif pos == 'right_leg': # ok
        pos_list = (1,8,11)
    elif pos == 'right_arm_n':
        pos_list = (1,5,6)
    elif pos == 'left_arm_n':
        pos_list = (1,2,3)
    # elif pos == 'right_hand':
    #     pos_list = (1,2,4)
    # elif pos == 'right_leg': # 180-angle 
    #     pos_list = (1,8,11)
    # # elif pos == 'right_ankle':
    # #     pos_list = (2,9,11)
    # elif pos == 'left_leg':
    #     pos_list = (1,8,14)
    else:
        logger.error('Unknown  [%s]', pos)
        return pnts

    for i in range(3):
        if human[pos_list[i]][2] <= 0.1:
            print('component [%d] incomplete'%(pos_list[i]))
            return pnts

        pnts.append((int( human[pos_list[i]][0]), int( human[pos_list[i]][1])))
    return pnts

# def get_angle_point(human, pos):
#     pnts = []

#     if pos == 'left_elbow':
#         pos_list = (5,6,7)
#     elif pos == 'left_hand':
#         pos_list = (1,5,7)
#     elif pos == 'left_knee':
#         pos_list = (12,13,14)
#     elif pos == 'left_ankle':
#         pos_list = (5,12,14)
#     elif pos == 'right_elbow':
#         logger.error('Unknown  [%s]', pos)
#         return pnts

#     for i in range(3):
#         if human[pos_list[i]][2] <= 0.1:
#             print('component [%d] incomplete'%(pos_list[i]))
#             return pnts

#         pnts.append((int( human[pos_list[i]][0]), int( human[pos_list[i]][1])))
#     return pnts

def angle_left_leg(human):
    pnts = get_angle_point(human, 'left_leg')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = 180-angle_between_points(pnts[0], pnts[1], pnts[2])
        # logger.info('left leg angle:%f'%(angle))
    return angle

def angle_right_leg(human):
    pnts = get_angle_point(human, 'right_leg')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = 180-angle_between_points(pnts[0], pnts[1], pnts[2])
        # logger.info('right leg angle:%f'%(angle))
    return angle

def angle_left_arm(human):
    pnts = get_angle_point(human, 'left_arm')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return -1, -1
    len_arm=math.sqrt((pnts[1][0]-pnts[2][0])**2+(pnts[1][1]-pnts[2][1])**2)
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        # logger.info('left arm angle:%f'%(angle))
    return angle, len_arm

def angle_right_arm(human):
    pnts = get_angle_point(human, 'right_arm')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return -1, -1
    len_arm=math.sqrt((pnts[1][0]-pnts[2][0])**2+(pnts[1][1]-pnts[2][1])**2)
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        # logger.info('right arm angle:%f'%(angle))
    return angle, len_arm

######

def angle_left_arm_n(human):
    pnts = get_angle_point(human, 'left_arm_n')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        # logger.info('left arm angle:%f'%(angle))
    return angle


def angle_right_arm_n(human):
    pnts = get_angle_point(human, 'right_arm_n')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        # logger.info('right arm angle:%f'%(angle))
    return angle

########

def angle_left_hand(human):
    pnts = get_angle_point(human, 'left_hand')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return -1

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('left hand angle:%f'%(angle))
    return angle


def angle_left_elbow(human):
    pnts = get_angle_point(human, 'left_elbow')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:

        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('left elbow angle:%f'%(angle))
    return angle

def angle_left_knee(human):
    pnts = get_angle_point(human, 'left_knee')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('left knee angle:%f'%(angle))
    return angle

def angle_left_ankle(human):
    pnts = get_angle_point(human, 'left_ankle')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('left ankle angle:%f'%(angle))
    return angle

def angle_right_hand(human):
    pnts = get_angle_point(human, 'right_hand')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('right hand angle:%f'%(angle))
    return angle


def angle_right_elbow(human):
    pnts = get_angle_point(human, 'right_elbow')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('right elbow angle:%f'%(angle))
    return angle

def angle_right_knee(human):
    pnts = get_angle_point(human, 'right_knee')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('right knee angle:%f'%(angle))
    return angle

def angle_right_ankle(human):
    pnts = get_angle_point(human, 'right_ankle')
    if len(pnts) != 3:
        logger.info('component incomplete')
        return

    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        logger.info('right ankle angle:%f'%(angle))
    return angle
# def cos_sim(A, B):
#     dot(A, B)/(norm(A)*norm(B)) # =cosx

#     return dot(A, B)/(norm(A)*norm(B))

# # 0을 기준으로 하는 벡터?
# doc1=np.array([0,1])
# doc2=np.array([1,0])

# # 정확도가 Thre 이상일 때, 각도에 대한 얘기
# # print(degrees(acos(cos_sim(doc1,doc2))))

# # print(math.degrees(math.acos(-1)))
# # ================================ 라이브러리 불러오기 및 openpose 시작 구조 ==============================

# # tensorflow warning not show
# Flags

# parser = argparse.ArgumentParser()
# parser.add_argument('--model_folder', type=str, default='../openpose/models/')
# parser.add_argument('--target_video', type=str, default='/home/piai/CV-pose-detection/media/tabata.mp4') # 빵느 영상
# parser.add_argument('--net_resolution', type=str, default='176x176')
# parser.add_argument('--cam_width', type=int, default=600)
# parser.add_argument('--cam_height', type=int, default=340)
# parser.add_argument('--number_people_max', type=int, default=1)
# args = parser.parse_args()

# # Custom openpose params
# params = dict()
# params['model_folder'] = args.model_folder
# params['net_resolution'] = args.net_resolution
# params['number_people_max'] = args.number_people_max
# params['display'] = 0
# params['disable_multi_thread'] = True

# # Start openpose
# opWrapper = op.WrapperPython()
# opWrapper.configure(params)
# opWrapper.start()

# pose = cv2.imread('/home/piai/CV-pose-detection/media/chest_open.png')
# webcam = cv2.imread('/home/piai/CV-pose-detection/media/stand.jpg')

# pose = crop_targetimage(pose, args.cam_width,args.cam_height)
# webcam_img = crop_targetimage(webcam, args.cam_width, args.cam_height)

# pose_datum = label_img(opWrapper, pose)
# webcam_datum = label_img(opWrapper, webcam_img)

# pose_coords_vec = make_vector(pose_datum.poseKeypoints)
# coords_vec = make_vector(webcam_datum.poseKeypoints)
# keypoint_list=webcam_datum.poseKeypoints
# print(webcam_datum.poseKeypoints)

# print(webcam_datum.NOSE)    

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

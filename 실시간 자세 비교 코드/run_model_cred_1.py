# ================================ 라이브러리 불러오기 및 openpose 시작 구조 ==============================

# tensorflow warning not show
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 모듈 import
import sys
import cv2
import argparse
import numpy as np
import time
from cal_angle import *
from keras.models import load_model
from util import *

sys.path.append('/usr/local/python')

from openpose import pyopenpose as op

# Flags
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='../openpose/models/')
parser.add_argument('--target_video', type=str, default='/home/piai/CV-pose-detection/media/tabata.mp4') # 빵느 영상
parser.add_argument('--net_resolution', type=str, default='176x176')
parser.add_argument('--cam_width', type=int, default=600)
parser.add_argument('--cam_height', type=int, default=340)
parser.add_argument('--number_people_max', type=int, default=1)

args = parser.parse_args()

# Custom openpose params
params = dict()
params['model_folder'] = args.model_folder
params['net_resolution'] = args.net_resolution
params['number_people_max'] = args.number_people_max
params['display'] = 0
params['disable_multi_thread'] = True

# Start openpose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

model = load_model('sampling_total_coord_with_yoga_32.h5')

# 딱 그 포즈와만 계속해서 비교해보자!!!! -> 모델 신뢰성 확인!

# ================================ 기준 포즈와 다른 포즈의 정확도 비교 ==============================

###### chest_open 자세와 chest_open 자세 비교 ###### 
pose = cv2.imread('/home/piai/CV-pose-detection/media/cross/squat.png')
webcam = cv2.imread('/home/piai/CV-pose-detection/media/cross/squat_per_n.jpg')

pose = crop_targetimage(pose, args.cam_width,args.cam_height)
webcam_img = crop_targetimage(webcam, args.cam_width, args.cam_height)

pose_datum = label_img(opWrapper, pose)
webcam_datum = label_img(opWrapper, webcam_img)

screen_out1 = pose
screen_out2 = webcam_datum.cvOutputData

#print('=================================')
angle_R_arm=angle_right_arm(webcam_datum.poseKeypoints[1] )
angle_L_arm=angle_left_arm(webcam_datum.poseKeypoints[1] )
angle_R_leg=angle_right_leg(webcam_datum.poseKeypoints[1] )
angle_L_leg=angle_left_leg(webcam_datum.poseKeypoints[1] )

# angle_R_arm=angle_right_arm_n(webcam_datum.poseKeypoints[i] )
# angle_L_arm=angle_left_arm_n(webcam_datum.poseKeypoints[i] )
# angle_R_leg=angle_right_leg(webcam_datum.poseKeypoints[i] )
# angle_L_leg=angle_left_leg(webcam_datum.poseKeypoints[i] )

angle_R_arm_f=angle_right_arm(pose_datum.poseKeypoints[1] )
angle_L_arm_f=angle_left_arm(pose_datum.poseKeypoints[1] )
angle_R_leg_f=angle_right_leg(pose_datum.poseKeypoints[1] )
angle_L_leg_f=angle_left_leg(pose_datum.poseKeypoints[1] )

pose_coords_vec = make_vector(pose_datum.poseKeypoints)
coords_vec = make_vector(webcam_datum.poseKeypoints)

input_vec = np.concatenate([pose_coords_vec, coords_vec]).flatten()
similarity_score = model.predict(input_vec.reshape((1, -1)))
print('chest_open 자세와 chest_open 자세(같은 자세 사이의) 유사도: {}'.format(similarity_score))

if accuracy >= 80 :
    feed = ''
else:
    if (angle_R_arm > 20 + angle_R_arm_f) and (angle_L_arm > 20 + angle_L_arm_f):
        feed = 'down arm'
    elif angle_R_arm > 20 + angle_R_arm_f:
        feed = 'down right arm'
    elif angle_R_arm < angle_R_arm_f - 20:
        feed = 'up right arm'
    elif angle_L_arm > 20 + angle_L_arm_f:
        feed = 'down left arm'
    elif angle_L_arm < angle_L_arm_f - 20:
        feed = 'up left arm'
    else:
        pass
cv2.putText(screen_out2, feed, (355, 355), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)
addh = cv2.hconcat([screen_out1, screen_out2])
cv2.imshow("WoW B4", addh)
# ###### chest_open 자세와 stand 자세 비교 ###### 
# pose = cv2.imread('/home/piai/CV-pose-detection/media/cross/squat.png')
# webcam = cv2.imread('/home/piai/CV-pose-detection/media/cross/squat_g_n2.jpg')

# pose = crop_targetimage(pose, args.cam_width,args.cam_height)
# webcam_img = crop_targetimage(webcam, args.cam_width, args.cam_height)

# pose_datum = label_img(opWrapper, pose)
# webcam_datum = label_img(opWrapper, webcam_img)

# pose_coords_vec = make_vector(pose_datum.poseKeypoints)
# coords_vec = make_vector(webcam_datum.poseKeypoints)

# input_vec = np.concatenate([pose_coords_vec, coords_vec]).flatten()
# similarity_score = model.predict(input_vec.reshape((1, -1)))
# print('chest_open 자세와 stand 자세(비슷한 자세 사이의) 유사도: {}'.format(similarity_score))

# ###### chest_open 자세와 스쿼트 자세 비교 ###### 
# pose = cv2.imread('/home/piai/CV-pose-detection/media/cross/squat.png')
# webcam = cv2.imread('/home/piai/CV-pose-detection/media/cross/squat_miss_n.jpg')

# pose = crop_targetimage(pose, args.cam_width,args.cam_height)
# webcam_img = crop_targetimage(webcam, args.cam_width, args.cam_height)

# # addh = cv2.hconcat([pose, webcam_img])
# # cv2.imshow("WoW B4", addh)
# # cv2.waitKey(0)

    addh = cv2.hconcat([screen_out1, screen_out2])

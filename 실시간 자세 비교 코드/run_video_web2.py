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
from keras.models import load_model
from util import *
from PIL import ImageFont, ImageDraw, Image

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QIcon, QImage, QPalette, QBrush,QColor,QFont
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import QSize, QTimer
from PyQt5 import QtCore
from scoreGUI import App

sys.path.append('/usr/local/python')

from openpose import pyopenpose as op
from sklearn.metrics.pairwise import cosine_similarity

# from playsound import playsound
from ffpyplayer.player import MediaPlayer
import signal

#Allows for the closing of PyQt
signal.signal(signal.SIGINT,signal.SIG_DFL)

import pafy


# Flags
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='../openpose/models/')
parser.add_argument('--target_video', type=str, default='/home/piai/CV-pose-detection/media/tabata.mp4') # 빵느 영상
# parser.add_argument('--target_video', type=str, default='/home/piai/CV-pose-detection/media/chest_open.mp4') # 빵느 영상
parser.add_argument('--net_resolution', type=str, default='176x176') # 해상도 16의 배수로 설정 -> 얘를 바꾸니깐 영상 저장이 제대로 안됨,,, 왜?ㅠ
parser.add_argument('--cam_width', type=int, default=960) # 화면사이즈 달라지면 accuracy 원 나오는 곳이 달라짐 -> puttext2로 만들어서 해결! (04/11)
parser.add_argument('--cam_height', type=int, default=540)
parser.add_argument('--number_people_max', type=int, default=1)

args = parser.parse_args()

# Custom openpose params
params = dict()
params['face'] = False
params['hand'] = False
params['model_folder'] = args.model_folder
params['net_resolution'] = args.net_resolution
params['number_people_max'] = args.number_people_max
params['display'] = 0
params['disable_multi_thread'] = True

# Start openpose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()



# =========== 여기는 기준 자세들로부터 벡터를 만드는 코드인데 연산량이 많아져서 미리 계산해서 넣어둠 =============


# 기준 자세들
# pose1 = cv2.imread('/home/piai/CV-pose-detection/media/chest_open.png')
# pose2 = cv2.imread('/home/piai/CV-pose-detection/media/chest_open2.png')
# pose3 = cv2.imread('/home/piai/CV-pose-detection/media/cross.png')
# pose4 = cv2.imread('/home/piai/CV-pose-detection/media/cross2.png')
# pose5 = cv2.imread('/home/piai/CV-pose-detection/media/narrow_1.')


# 기준 이미지
# pose1 = crop_targetimage(pose1, args.cam_width,args.cam_height)
# pose_datum1 = label_img(opWrapper, pose1)
# pose_coords_vec1 = make_vector(pose_datum1.poseKeypoints)

# pose2 = crop_targetimage(pose2, args.cam_width,args.cam_height)
# pose_datum2 = label_img(opWrapper, pose2)
# pose_coords_vec2 = make_vector(pose_datum2.poseKeypoints)

# pose3 = crop_targetimage(pose3, args.cam_width,args.cam_height)
# pose_datum3 = label_img(opWrapper, pose3)
# pose_coords_vec3 = make_vector(pose_datum3.poseKeypoints)

# pose4 = crop_targetimage(pose4, args.cam_width,args.cam_height)
# pose_datum4 = label_img(opWrapper, pose4)
# pose_coords_vec4 = make_vector(pose_datum4.poseKeypoints)

# pose5 = crop_targetimage(pose5, args.cam_width,args.cam_height)
# pose_datum5 = label_img(opWrapper, pose5)
# pose_coords_vec5 = make_vector(pose_datum5.poseKeypoints)
# print([pose_coords_vec5])

# ================================ 운동 영상, 웹캠 영상 받아오기 ==============================

# Start streams
# target : 유튜브 강사 영상
# webcam : webcam 영상
'''유튜브 url로 동영상 가져오기'''
url = 'https://www.youtube.com/watch?v=ly_oclMG4SE'
# url = 'https://www.youtube.com/watch?v=ZbN8b4_Nh3w'
video = pafy.new(url)
print('title = ', video.title)
# print('video.rating = ', video.rating)
# print('video.duration = ', video.duration)

best = video.getbest(preftype='mp4')     # 'webm','3gp'
print('best.resolution', best.resolution)
 
cap=cv2.VideoCapture(best.url)
player = MediaPlayer(args.target_video) # target 오디오 저장

webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture('/home/piai/CV-pose-detection/media/correct.mp4')
# webcam = cv2.VideoCapture('/home/piai/CV-pose-detection/media/compare_chestopen.mp4')
# webcam = cv2.VideoCapture(best.url)

# 동영상 저장 코덱
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 영상 저장 크기가 안맞아서 에러 뜸 -> 맞춰줘서 해결! (04/11)
w = args.cam_width
w_min = w // 2 - (w // 4)
w_max = w // 2 + (w // 4)
w_out = w_max - w_min

# 영상 저장하는 코드
out = cv2.VideoWriter(
    'standard_user_second_output.mp4',
    fourcc,
    20.0, # 저장할 frame 수
    (w_out, args.cam_height) # cam_width: 1280, cam_height: 720
)

# Setup framerate params
count = 0
frames = 0
start = time.time()
# time.sleep(2)  # delay to wait for detection
# model = load_model('ComparatorNet.h5')
model = load_model('model.h5') # 우리 비디오로 새로 만든 모델 (04/10)


# ================================ 기준 자세들을 미리 저장 START ==============================

# chest open1
pose_coords_vec1 = [[[ 8.4107406e-02, -3.1943253e-01,  9.4386905e-01],
  [ 8.5312553e-02, -2.4707897e-01,  9.6523243e-01],
  [ 1.9782271e-02, -2.5467640e-01,  9.6682394e-01],
  [-9.5008440e-02, -4.0208355e-01,  9.1066033e-01],
  [-1.5062402e-01, -6.6072088e-01,  7.3536402e-01],
  [ 1.6286422e-01, -2.6586369e-01,  9.5015359e-01],
  [ 4.7346008e-01, -4.9165392e-01,  7.3082966e-01],
  [-8.2084399e-01, -5.7115251e-01,  0.0000000e+00],
  [ 9.3811922e-02, -1.1437466e-02,  9.9552429e-01],
  [ 3.3662003e-02, -2.4710424e-02,  9.9912775e-01],
  [-5.1689330e-02,  2.0281754e-01,  9.7785139e-01],
  [-1.2949701e-01,  3.6072013e-01,  9.2364043e-01],
  [ 1.3206573e-01,  2.7436338e-04,  9.9124092e-01],
  [ 1.0126613e-01,  2.0819999e-01,  9.7282988e-01],
  [ 9.0161473e-02,  4.2422345e-01,  9.0105790e-01],
  [ 6.8262130e-02, -3.4209216e-01,  9.3718368e-01],
  [ 8.4793679e-02, -3.2993847e-01,  9.4018644e-01],
  [ 3.4958843e-02, -4.0215814e-01,  9.1490257e-01],
  [ 1.1919102e-01, -3.8110113e-01,  9.1681814e-01],
  [ 1.0028209e-01,  4.8132741e-01,  8.7078553e-01],
  [ 1.1244717e-01,  4.7666457e-01,  8.7186384e-01],
  [ 5.6050897e-02,  4.9417931e-01,  8.6755121e-01],
  [-1.4523825e-01,  4.5464504e-01,  8.7875128e-01],
  [-1.8291742e-01,  4.6768564e-01,  8.6476094e-01],
  [-1.4675987e-01,  4.0974054e-01,  9.0031892e-01]]]

# chest open2
pose_coords_vec2 = [[[-0.14652327, -0.34215486,  0.9281492 ],
    [-0.15201923, -0.2560349 ,  0.9546394 ],
    [-0.31944728, -0.26045159,  0.9111083 ],
    [-0.37963045, -0.42574403,  0.82135427],
    [-0.27531138, -0.7052191 ,  0.6533526 ],
    [ 0.03705043, -0.26623812,  0.96319497],
    [ 0.19016036, -0.43101546,  0.8820798 ],
    [ 0.11816317, -0.6118861 ,  0.7820696 ],
    [-0.17110641, -0.00358218,  0.98524606],
    [-0.24262197,  0.01033223,  0.9700659 ],
    [-0.18908055,  0.25114828,  0.94930136],
    [-0.15131608,  0.46746168,  0.87096673],
    [ 0.04172243, -0.0179831 ,  0.9989674 ],
    [ 0.25964934,  0.23131886,  0.93758935],
    [ 0.42319074,  0.3767952 ,  0.82397515],
    [-0.15496883, -0.3680233 ,  0.91681165],
    [-0.0766755 , -0.39325723,  0.91622573],
    [-0.26324296, -0.44728127,  0.8547763 ],
    [ 0.01590662, -0.5677767 ,  0.823029  ],
    [ 0.49413863,  0.47503588,  0.72812635],
    [ 0.58622015,  0.44435665,  0.67741656],
    [ 0.46714258,  0.42793643,  0.77372354],
    [-0.13124536,  0.55235076,  0.8232152 ],
    [-0.18465072,  0.5410294 ,  0.8204824 ],
    [-0.10829806,  0.54032904,  0.83445555]]]

# squat
pose_coords_vec3 = [[[ 0.11418229, -0.4966258 ,  0.8604216 ],
        [ 0.11439768, -0.44484755,  0.88827014],
        [ 0.0237209 , -0.4729484 ,  0.8807708 ],
        [-0.00297683, -0.6349746 ,  0.77252734],
        [ 0.04889262, -0.49606457,  0.86690795],
        [ 0.15500161, -0.3327992 ,  0.93017155],
        [ 0.23031516, -0.35189655,  0.90726167],
        [ 0.18985197, -0.4888171 ,  0.85147756],
        [ 0.06283852, -0.06881052,  0.99564874],
        [-0.00197265, -0.10694247,  0.99426335],
        [-0.00262249,  0.69816816,  0.7159289 ],
        [ 0.0216029 ,  0.81665176,  0.57672626],
        [ 0.05370756,  0.04565542,  0.99751246],
        [ 0.01938657,  0.51869565,  0.8547391 ],
        [ 0.01814314,  0.6834117 ,  0.72980773],
        [ 0.07943787, -0.54111826,  0.83718616],
        [ 0.14677033, -0.49620214,  0.8557114 ],
        [ 0.06333091, -0.8376014 ,  0.5425984 ],
        [ 0.20609857, -0.61897796,  0.7578851 ],
        [-0.02412815,  0.8933173 ,  0.4487784 ],
        [-0.00163078,  0.8801009 ,  0.474784  ],
        [ 0.03728649,  0.76840794,  0.63887316],
        [-0.02590329,  0.970238  ,  0.24076416],
        [-0.8563127 , -0.5164578 ,  0.        ],
        [ 0.0468325 ,  0.899859  ,  0.43365934]]]

# cross1
pose_coords_vec4 = [[[-0.35416234, -0.5672679 ,  0.7434891 ],
        [ 0.18680935, -0.36893123,  0.91049004],
        [-0.4093672 , -0.33771718,  0.84756446],
        [-0.5765421 , -0.27302504,  0.77010167],
        [-0.38653198, -0.47365987,  0.7913529 ],
        [ 0.30822885, -0.40210596,  0.86215186],
        [ 0.47695193, -0.3312222 ,  0.81413066],
        [-0.25261226, -0.45629713,  0.8532174 ],
        [ 0.34161776,  0.03623917,  0.93914   ],
        [ 0.10885607,  0.03695549,  0.9933703 ],
        [ 0.5418375 ,  0.40496293,  0.7364897 ],
        [ 0.30630806,  0.85085505,  0.4268735 ],
        [ 0.54579955,  0.03382714,  0.8372327 ],
        [ 0.5283859 ,  0.38993865,  0.7541593 ],
        [ 0.14826693,  0.80975217,  0.56773096],
        [-0.48283616, -0.5873938 ,  0.6494904 ],
        [-0.24158317, -0.57242405,  0.7835613 ],
        [-0.6773027 , -0.6650407 ,  0.3146141 ],
        [ 0.10802148, -0.6356364 ,  0.7643937 ],
        [-0.32420158,  0.89511985,  0.3060289 ],
        [ 0.14153636,  0.9443405 ,  0.29696527],
        [-0.01573502,  0.8717878 ,  0.4896309 ],
        [ 0.14111151,  0.96153337,  0.23567156],
        [ 0.14273944,  0.96796495,  0.2065659 ],
        [ 0.28962433,  0.89834017,  0.33030686]]]

# cross2
pose_coords_vec5 = [[[-0.00199469, -0.43926203,  0.8983568 ],
        [-0.00205247, -0.399675  ,  0.9166546 ],
        [-0.18318892, -0.39323705,  0.901003  ],
        [-0.47739556, -0.26953325,  0.8363285 ],
        [-0.24451308, -0.44037205,  0.86387837],
        [ 0.21419522, -0.38442755,  0.8979621 ],
        [ 0.4519952 , -0.27284655,  0.84926736],
        [ 0.2826953 , -0.41628256,  0.8641713 ],
        [-0.0024973 ,  0.0498962 ,  0.9987513 ],
        [-0.15074448,  0.0461419 ,  0.9874953 ],
        [-0.35756248,  0.1835909 ,  0.9156656 ],
        [-0.32663855,  0.4438537 ,  0.8344466 ],
        [ 0.12106144,  0.04813613,  0.9914772 ],
        [ 0.37449148,  0.17320907,  0.91090876],
        [ 0.3137831 ,  0.46578452,  0.82739645],
        [-0.08404087, -0.44177482,  0.8931809 ],
        [ 0.05269991, -0.44226223,  0.89533615],
        [-0.121102  , -0.4710103 ,  0.8737755 ],
        [ 0.13870265, -0.5710175 ,  0.8091357 ],
        [ 0.43035594,  0.48830384,  0.75917935],
        [ 0.45186427,  0.4835811 ,  0.7496453 ],
        [ 0.29407316,  0.49714726,  0.81631213],
        [-0.3705368 ,  0.52422667,  0.7667392 ],
        [-0.4488946 ,  0.5222553 ,  0.7250814 ],
        [-0.3275195 ,  0.5007671 ,  0.8012262 ]]]

# narrow squat
pose_coords_vec6 = [[[ 0.16966163, -0.16061829,  0.9723254 ],
        [ 0.09929433, -0.0945971 ,  0.9905514 ],
        [ 0.04158531, -0.10051829,  0.9940658 ],
        [ 0.07287188,  0.01422232,  0.9972399 ],
        [ 0.13158128, -0.07448155,  0.98850334],
        [ 0.17421085, -0.08715871,  0.9808435 ],
        [ 0.25542796,  0.01345879,  0.9667344 ],
        [ 0.19972402, -0.0767643 ,  0.97684056],
        [ 0.03300608,  0.11051971,  0.9933257 ],
        [ 0.01017046,  0.1067318 ,  0.9942359 ],
        [ 0.10801445,  0.17326987,  0.9789333 ],
        [ 0.08841478,  0.35717857,  0.92984205],
        [ 0.10215541,  0.11497842,  0.9881013 ],
        [ 0.15702453,  0.31423163,  0.9362701 ],
        [ 0.12174176,  0.533857  ,  0.83676505],
        [ 0.12882428, -0.16988975,  0.97700655],
        [ 0.18100852, -0.17289755,  0.96816444],
        [ 0.09665251, -0.16629393,  0.981328  ],
        [-0.7866884 , -0.61735034,  0.        ],
        [ 0.23668295,  0.77819926,  0.5817104 ],
        [-0.7866884 , -0.61735034,  0.        ],
        [-0.7866884 , -0.61735034,  0.        ],
        [ 0.14183769,  0.45251772,  0.8804032 ],
        [ 0.11189064,  0.43425146,  0.8938155 ],
        [ 0.05418651,  0.41663188,  0.90745896]]]


# ================================ 기준 자세들을 미리 저장 END ==============================

# pose_coords = [pose_coords_vec1, pose_coords_vec2]
# pose_coords = [pose_coords_vec1, pose_coords_vec2, pose_coords_vec3, pose_coords_vec4, pose_coords_vec5]

# pose = cv2.imread('/home/piai/CV-pose-detection/media/chest_open.png')
# pose = crop_targetimage(pose, args.cam_width,args.cam_height)
# pose_datum = label_img(opWrapper, pose)
# pose_coords_vec = make_vector(pose_datum.poseKeypoints)


# ================================ 기준 이미지와 웹캠 영상 좌표 비교하기 START ==============================

# 여기서 while문을 돌면서 계속해서 영상 프레임을 비교하니깐 연산량이 많아지는 거 아닐까...?
# 그럼 While True로 계속 돌지말고 어떤 이벤트(예를 들면 딱 그 동작을 했을 때만 비교하면 연산량이 줄어들겠지...?)
    
# target_img = get_target_image(target, args.cam_width, args.cam_height)

# ================================ 트랙바 만들기 ==============================
# def getFrame(frame_nr):
#     global target
#     target.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)

# nr_of_frames = int(target.get(cv2.CAP_PROP_FRAME_COUNT))
# cv2.createTrackbar("Frame", "WoW", 0,nr_of_frames,getFrame)
# # chest_open1_frame = []
# for i in range(25):
#     temp = 625 + i*25
#     chest_open1_frame.append(temp-1)
#     chest_open1_frame.append(temp)
#     chest_open1_frame.append(temp+1)
# # print(chest_open1_frame)

# chest_open2_frame = []
# for i in range(25):
#     chest_open2_frame.append(650 + i*50)
# print(chest_open2_frame)

# # ordinal_score 초기화
ordinal_score = (0, '', 0.0, (0, 0, 0))
similarity_score_list = []
font = cv2.FONT_HERSHEY_COMPLEX 



while True:
    count += 1
    frames += 1

    # Get images
    frame_img = get_target_image(cap, args.cam_width, args.cam_height)
    webcam_img = get_image(webcam, args.cam_width, args.cam_height)
    # Label images
    webcam_datum = label_img(opWrapper, webcam_img)

    # 오디오 불러오기
    audio_frame, val = player.get_frame()

    # output할 화면 설정
    screen_out1 = frame_img
    # screen_out1 = target_img
    screen_out2 = webcam_datum.cvOutputData

    # Add overlay to show results
    # overlay = screen_out2.copy()
    # cv2.rectangle(overlay, (0, 0), (args.cam_width, args.cam_height),
    #             ordinal_score[3], 2)

    # Draw a vertical white line with thickness of 10 px
    cv2.line(screen_out1, (args.cam_width, 0),
            (args.cam_width, args.cam_height),
            (255, 255, 255), 10)
            

    # 1.유사도를 비교해야하는 그 프레임에서만 비교!!
    # if count in chest_open1_frame:
    #     print(count)

    # if (count % 25 == 0) :

    # 2. 운동을 시작하는 순간부터 1초마다 비교!
    # 유사도를 리스트 안에 넣어두고 1초마다 그 평균을 계산해서 output으로 내보냄

    # 기준 좌표 설정
    # 1분 14초 까지는 chest open
    # if count // 25 < 75:
    #     pose_coords_vec = pose_coords_vec1
    # elif 75 <= count // 25 <= 83:
    #     ordinal_score = (0, '', 0.0, (0, 0, 0))
    # # 2분 14초 까지는 squat & cross
    # elif 83 < count // 25 <= 134:
    #     pose_coords_vec = pose_coords_vec5




    if (count > 624) & (count % 25 == 0):
    # # if count % 10 == 0:
    #     # ========================== 포즈 유사도 비교하는 부분 =========================
    #     # Check if OpenPose managed to label
        if type(webcam_datum.poseKeypoints) == np.ndarray and \
            webcam_datum.poseKeypoints.shape == (1, 25, 3):
            # Scale, transform, normalize, reshape, predict

            # list에 넣은 거
            # coords_vec = make_vector(webcam_datum.poseKeypoints)
            # score_temp = []
            # for pose_coords_vec in pose_coords:
            #     input_vec = np.concatenate([pose_coords_vec, coords_vec]).flatten()
            #     similarity_score = model.predict(input_vec.reshape((1, -1)))
            #     score_temp.append(similarity_score)
            # ordinal_score = get_ordinal_score(max(score_temp))
        # ordinal_score = (0, '', 0.0, (0, 0, 0))
        # if type(frame_datum.poseKeypoints) != type(None):
        #     if type(webcam_datum.poseKeypoints) == np.ndarray and \
        #         webcam_datum.poseKeypoints.shape == (1, 25, 3):

        #         if type(frame_datum.poseKeypoints) == np.ndarray or \
        #             frame_datum.poseKeypoints.shape == (1, 25, 3):

        #             # Scale, transform, normalize, reshape, predict
                    
        #             coords_vec = make_vector(webcam_datum.poseKeypoints)
        #             target_coords_vec = make_vector(frame_datum.poseKeypoints)
        #             input_vec = np.concatenate([coords_vec, target_coords_vec]).flatten()
        #             similarity_score = model.predict(input_vec.reshape((1, -1))) # -> 여기가 문제다~ 속도가 느려진다~
        #             ordinal_score = get_ordinal_score(similarity_score)        
        #             cv2.putText(screen_out2, ' ' + ordinal_score[1], (355, 145), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)
            
            coords_vec = make_vector(webcam_datum.poseKeypoints)
            input_vec = np.concatenate([pose_coords_vec1, coords_vec]).flatten()
            similarity_score = model.predict(input_vec.reshape((1, -1)))
            ordinal_score = get_ordinal_score(similarity_score)
            # similarity_score_list.append(similarity_score)
            # if count % 25 == 0:
            # if count in [635, 685, 735, 785, 835, 885, 945, 995, 1045, 1095, 1145, 1195, 1245, 1295, 1345, 1395, 1445, 1495, 1545, 1595, 1645, 1695, 1745, 1795, 1845 ] :
                # ordinal_score = get_ordinal_score(max(similarity_score_list))
                # similarity_score_list = [] # 다시 리스트 초기화
                # cv2.putText(screen_out2, ' ' + ordinal_score[1], (355, 145), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)

            # elif count // 25 <= 74:
            #     pose_coords_vec = pose_coords_vec2
            #     coords_vec = make_vector(webcam_datum.poseKeypoints)
            #     input_vec = np.concatenate([pose_coords_vec, coords_vec]).flatten()
            #     similarity_score = model.predict(input_vec.reshape((1, -1))) # -> 여기가 문제다~ 속도가 느려진다~
            #     similarity_score_list.append(similarity_score)
            #     if count % 25 == 0:
            #         ordinal_score = get_ordinal_score(max(similarity_score))
            #         similarity_score_list = [] # 다시 리스트 초기화

            # elif 74 < count // 25 <= 83:
            #     ordinal_score = (0, '', 0.0, (0, 0, 0))

            # # 2분 14초 까지는 squat & cross
            # elif 83 < count // 25 <= 134:
            #     pose_coords_vec = pose_coords_vec5
            #     coords_vec = make_vector(webcam_datum.poseKeypoints)
            #     input_vec = np.concatenate([pose_coords_vec, coords_vec]).flatten()
            #     similarity_score = model.predict(input_vec.reshape((1, -1))) # -> 여기가 문제다~ 속도가 느려진다~
            #     similarity_score_list.append(similarity_score)
            #     if count % 50 == 0:
            #         ordinal_score = get_ordinal_score(max(similarity_score))
            #         similarity_score_list = [] # 다시 리스트 초기화

    # 휴식시간
    elif 75 <= count // 25 <= 83:
        ordinal_score = (0, '', 0.0, (0, 0, 0))

    # Display comment
    accuracy = int(ordinal_score[0]*100)
    # Perfect
    if accuracy >= 90 :
        puttext2(screen_out2, ordinal_score[0], ordinal_score[1])
    # Good
    elif accuracy >= 60 & accuracy < 90 :
        puttext2(screen_out2, ordinal_score[0], ordinal_score[1])
    # Miss
    else: 
        puttext2(screen_out2, ordinal_score[0], ordinal_score[1])
        cv2.putText(screen_out2, 'Please match the Timing!', (300, 145), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)
        # cv2.putText(screen_out2, ' ' + score2, (365, 145), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)
    
    # font = cv2.FONT_HERSHEY_COMPLEX 
    # accuracy = str(int(score1*100))
    # cv2.putText(screen_out2, accuracy, (392, 70), font, 0.7, (205, 255, 15), 2, cv2.LINE_AA)
    # cv2.putText(screen_out2, 'ACCURACY', (360, 95), font, 0.5, (205, 255, 15), 1, cv2.LINE_AA)
    # cv2.circle(screen_out2, (400,75), 45, (205, 255, 15), 2)

    # Record Video
    out.write(screen_out2)
    # Display img
    addh = cv2.hconcat([screen_out1, screen_out2])
    # cv2.imshow('', frame)
    cv2.imshow("WoW B4", addh)
    if val != 'eof' and audio_frame is not None:
        #audio
        img, t = audio_frame
    # Check for quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        # highscore = 99
        # # break
        # # highscore = round(current_score/2)
        # # if highscore % 5 != 0: 
        # #     highscore += (5 - highscore % 5)
        # #     print("Highscoreeeeeeeeeeeeeeeee",highscore)
        # #     # Opens leaderboard
        # app = QApplication(sys.argv)
        # ex = App(curr_score=highscore)
        # ex.show()
        # sys.exit(app.exec_())
        # ex.quit()
        print("hi")
        break


    # frame 수 cmd에 출력
    # Print frame rate : 1초마다 프레임 수
    if time.time() - start >= 1:
        # framerate = frames
        print('Frame: ', frames)
        frames = 0
        start = time.time()
    


# ================================ 기준 이미지와 웹캠 영상 좌표 비교하기 END ==============================

        
# Clean up
webcam.release()
# target.release()
out.release()
cv2.destroyAllWindows()
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
from cal_angle import *
from PIL import ImageFont, ImageDraw, Image
import logging

sys.path.append('/usr/local/python')

from openpose import pyopenpose as op
from sklearn.metrics.pairwise import cosine_similarity

# from playsound import playsound
from ffpyplayer.player import MediaPlayer
import signal

#Allows for the closing of PyQt
signal.signal(signal.SIGINT,signal.SIG_DFL)

import pafy

# from PyQt5.QtWidgets import QMainWindow, QApplication
# from PyQt5.QtGui import QIcon, QImage, QPalette, QBrush,QColor,QFont
# from PyQt5.QtCore import pyqtSlot
# from PyQt5.QtCore import QSize, QTimer
# from PyQt5 import QtCore
# from scoreGUI import App

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

# for calculaing angle
logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



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

target = cv2.VideoCapture(args.target_video) # target 비디오를 받아와서 frame 단위로 저장?
player = MediaPlayer(args.target_video) # target 오디오 저장

webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture('/home/piai/CV-pose-detection/media/correct2.mp4')
# webcam = cv2.VideoCapture('/home/piai/CV-pose-detection/media/wrong.MOV')

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
    'youtube_user_second_output.mp4',
    fourcc,
    20.0, # 저장할 frame 수
    (w_out, args.cam_height) # cam_width: 1280, cam_height: 720
)

# Setup framerate params
count = 0
frames = 0
start = time.time()
# time.sleep(2)  # delay to wait for detection
# model = load_model('ComparatorNet.h5') # yopoco 요가만 있는 모델
# model = load_model('model.h5') # 우리 비디오로 새로 만든 모델 (04/10)
model = load_model('model_total.h5') # 우리 비디오 + yopoco 로 새로 만든 모델  (04/10)
# model = load_model('model_final_64.h5') 
# model = load_model('model_final_with_yoga_32.h5')



# # ordinal_score 초기화
ordinal_score = (0, '', 0.0, (0, 0, 0))
similarity_score_list = []
font = cv2.FONT_HERSHEY_COMPLEX 
flag=0
state=''
while True:
    count += 1
    frames += 1

    # Get images
    frame_img = get_target_image(cap, args.cam_width, args.cam_height)
    webcam_img = get_image(webcam, args.cam_width, args.cam_height)
    # Label images
    frame_datum = label_img(opWrapper, frame_img)
    webcam_datum = label_img(opWrapper, webcam_img)
    # 오디오 불러오기
    audio_frame, val = player.get_frame()
    # output할 화면 설정
    screen_out1 = frame_img
    screen_out2 = webcam_datum.cvOutputData
    # Add overlay to show results
    # overlay = screen_out2.copy()
    # cv2.rectangle(overlay, (0, 0), (args.cam_width, args.cam_height),
    #             ordinal_score[3], 2)

    # Add overlay to show results
    overlay = screen_out2.copy()
    cv2.rectangle(overlay, (0, 0), (args.cam_width // 2, args.cam_height),
                  ordinal_score[3], 1)
    screen_out = cv2.addWeighted(overlay, ordinal_score[2],
                                 screen_out2,
                                 1 - ordinal_score[2], 0,
                                 screen_out2)
    
    
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

    angle_R_arm=0
    angle_L_arm=0
    angle_R_leg=0
    angle_L_leg=0
    angle_R_len=0
    angle_L_len=0
    if (count > 624) & (count % 25 == 0):
    # # if count % 10 == 0:
    #     # ========================== 포즈 유사도 비교하는 부분 =========================
    #     # Check if OpenPose managed to label
    #     if type(webcam_datum.poseKeypoints) == np.ndarray and \
    #         webcam_datum.poseKeypoints.shape == (1, 25, 3):
    #         # Scale, transform, normalize, reshape, predict

            # list에 넣은 거
            # coords_vec = make_vector(webcam_datum.poseKeypoints)
            # score_temp = []
            # for pose_coords_vec in pose_coords:
            #     input_vec = np.concatenate([pose_coords_vec, coords_vec]).flatten()
            #     similarity_score = model.predict(input_vec.reshape((1, -1)))
            #     score_temp.append(similarity_score)
            # ordinal_score = get_ordinal_score(max(score_temp))
        ordinal_score = (0, '', 0.0, (0, 0, 0))
        if type(frame_datum.poseKeypoints) != type(None):
            if type(webcam_datum.poseKeypoints) == np.ndarray and \
                webcam_datum.poseKeypoints.shape == (1, 25, 3):

                if type(frame_datum.poseKeypoints) == np.ndarray or \
                    frame_datum.poseKeypoints.shape == (1, 25, 3):
                    
                    # for calculating angle.
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    newImage = webcam_datum.cvOutputData[:, :, :]
                    human_count = len(webcam_datum.poseKeypoints)
                    for i in range(human_count):
                        for j in range(25):
                            if webcam_datum.poseKeypoints[i][j][2] > 0.01:
                                cv2.putText(newImage,str(j),  ( int(webcam_datum.poseKeypoints[i][j][0]) + 10,  int(webcam_datum.poseKeypoints[i][j][1])), font, 0.5, (0,255,0), 2) 
                    # cv2.imwrite(img_name, newImage)
                    # cv2.destroyAllWindows()      
                    
                    for i in range(human_count):
                        # print('=================================')
                        angle_R_arm, angle_R_len=angle_right_arm(webcam_datum.poseKeypoints[i] )
                        angle_L_arm, angle_L_len=angle_left_arm(webcam_datum.poseKeypoints[i] )
                        angle_R_leg=angle_right_leg(webcam_datum.poseKeypoints[i] )
                        angle_L_leg=angle_left_leg(webcam_datum.poseKeypoints[i] )
                    # Scale, transform, normaliz9:
                    #     puttext3(screen_out2, angle_R_arm,angle_L_arm,angle_R_leg,angle_L_leg)
                    coords_vec = make_vector(webcam_datum.poseKeypoints)
                    target_coords_vec = make_vector(frame_datum.poseKeypoints)
                    input_vec = np.concatenate([coords_vec, target_coords_vec]).flatten()
                    similarity_score = model.predict(input_vec.reshape((1, -1))) # -> 여기가 문제다~ 속도가 느려진다~
                    similarity_score_list.append(similarity_score)      
                    ordinal_score = get_ordinal_score(similarity_score)  
                    # if flag==2:
                    #     flag=0
                    # print('acc, angle_R_len, angle_L_len, angle_R_arm: ',similarity_score,angle_R_len, angle_L_len,angle_R_arm)
                    # if angle_R_len >=130 and angle_L_len >=130:
                    #     if angle_R_arm>=100: # 오른팔 내려 
                    #         if flag==0:
                    #             flag=1
                    #         else:
                    #             flag+=1
                    # cv2.putText(screen_out2, ' ' + ordinal_score[1], (355, 145), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)
                    # if similarity_score >= 0.6 and similarity_score < 0.9:
                    #     puttext3(screen_out2, angle_R_arm,angle_L_arm,angle_R_leg,angle_L_leg)


     # 휴식시간
    # elif 75 <= count // 25 <= 83:
    #     ordinal_score = (0, '', 0.0, (0, 0, 0))
    # 1초뒤에 어떻게 사라지게 하지?
    # Display comment
    accuracy = int(ordinal_score[0]*100)
    
    ###################각도 판단 알고리즘#######################
        # puttext3(screen_out2, angle_R_arm, angle_L_arm, angle_R_leg, angle_L_leg, angle_R_len, angle_L_len)

    if angle_R_len >=130 and angle_L_len >=130:
        if angle_R_arm>=100: 
            state='R_arm'
            end_count=0
            puttext3(screen_out2, angle_R_arm, angle_L_arm, angle_R_leg, angle_L_leg, angle_R_len, angle_L_len) # 2초뒤에 없어지게. 2초안에 또 생기면 다른 내용으로 업데이트
            past_angle_R_arm=angle_R_arm
            past_angle_L_arm=angle_L_arm
            past_angle_R_leg=angle_R_leg
            past_angle_L_leg=angle_L_leg
            past_angle_R_len=angle_R_len
            past_angle_L_len=angle_L_len
    # else:
    #     end_count <=30
    #     state=''
    
    if state=='R_arm':
        end_count+=1
        puttext3(screen_out2, past_angle_R_arm, past_angle_L_arm, past_angle_R_leg, past_angle_L_leg, past_angle_R_len, past_angle_L_len) 
        if end_count==30: # 30프레임 지나면 state 초기화
            # print(end_count)
            state=''
    ###################각도 판단 알고리즘#######################

    if accuracy >= 90 :
        # print(ordinal_score)
        puttext2(screen_out2, ordinal_score[0], ordinal_score[1])
        # if angle_R_arm>=100 or angle_L_arm 
    # Good
    elif accuracy >= 60 and accuracy < 90 :
        puttext2(screen_out2, ordinal_score[0], ordinal_score[1]) 
    # Miss
    # else: 
    #     puttext2(screen_out2, ordinal_score[0], ordinal_score[1])
    #     cv2.putText(screen_out2, 'Please match the Timing!', (300, 145), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)
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
        print("Average accuracy: {}".format(sum(similarity_score_list)/len(similarity_score_list)))
        break


    # frame 수 cmd에 출력
    # Print frame rate : 1초마다 프레임 수
    if time.time() - start >= 1:
        # framerate = frames
        print('Frame: ', frames)
        frames = 0
        start = time.time()
    


# ================================ 기준 이미지와 웹캠 영상 좌표 비교하기 END ==============================

        
# Clean upnumpy
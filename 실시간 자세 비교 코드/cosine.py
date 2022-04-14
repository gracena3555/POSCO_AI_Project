
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

# target = cv2.VideoCapture(args.target_video) # target 비디오를 받아와서 frame 단위로 저장?
player = MediaPlayer(args.target_video) # target 오디오 저장

webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture('/home/piai/CV-pose-detection/media/correct2.mp4')
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
ordinal_score = (0, '', 0.0, (0, 0, 0))
## run_video code
while True:
    frames+=1
    # Get images
    target_img = get_target_image(cap, args.cam_width, args.cam_height)
    # target_img = cv2.flip(target_img, 1)
    webcam_img = get_image(webcam, args.cam_width, args.cam_height)
    # Label images
    target_datum = label_img(opWrapper, target_img)
    webcam_datum = label_img(opWrapper, webcam_img)
    # ordinal_score = (0, '', 0.0, (0, 0, 0))
    # if (frames > 624) & (frames % 25 == 0):
    # if frames % 25 ==0: # 두 동작을 모두 비교해야함
    # ordinal_score = (0, '', 0.0, (0, 0, 0))
    # if type(webcam_datum.poseKeypoints) == np.ndarray and webcam_datum.poseKeypoints.shape == (1, 25, 3):
    if type(webcam_datum.poseKeypoints) == np.ndarray and type(target_datum.poseKeypoints) == np.ndarray and webcam_datum.poseKeypoints.shape == (1, 25, 3) and target_datum.poseKeypoints.shape == (1, 25, 3) :
        # Scale, transform, normalize, reshape, predict
        # coord_c=webcam_datum.poseKeypoints # vector 변환 전 cosine_score
        coords_vec = make_vector(webcam_datum.poseKeypoints) # vector 변환 후
        pose_coords_vec = make_vector(target_datum.poseKeypoints)
        vec2=scale_transform2(pose_coords_vec[0])
        vec1=scale_transform2(coords_vec[0]) # 이것도 1,50 ndarry 출력해야함 =>
        cosine_score = cosine_similarity(vec1, vec2)[0]
        ordinal_score = get_ordinal_score(cosine_score)
        puttext2(screen_out2, ordinal_score[0], ordinal_score[1])

    
    screen_out1 = target_img
    screen_out2 = webcam_datum.cvOutputData



# Display comment
    accuracy = int(ordinal_score[0]*100)
    puttext2(screen_out2, ordinal_score[0], ordinal_score[1])
    # Perfect
    # if accuracy >= 90 :
    #     puttext2(screen_out2, ordinal_score[0], ordinal_score[1])
    # # Good
    # elif accuracy >= 60 & accuracy < 90 :
    #     puttext2(screen_out2, ordinal_score[0], ordinal_score[1])
    # # Miss
    # else: 
    #     puttext2(screen_out2, ordinal_score[0], ordinal_score[1])
    #     cv2.putText(screen_out2, 'Please match the Timing!', (300, 145), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)
    #     # cv2.putText(screen_out2, ' ' + score2, (365, 145), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)
    
    # # font = cv2.FONT_HERSHEY_COMPLEX 
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
    # if val != 'eof' and audio_frame is not None:
    #     #audio
    #     img, t = audio_frame
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
    # if time.time() - start >= 1:
    #     # framerate = frames
    #     print('Frame: ', frames)
    #     frames = 0
    #     start = time.time()
    


# ================================ 기준 이미지와 웹캠 영상 좌표 비교하기 END ==============================

        
# Clean up
webcam.release()
# target.release()
out.release()
cv2.destroyAllWindows()
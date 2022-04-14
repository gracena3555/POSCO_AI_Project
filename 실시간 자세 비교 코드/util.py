import sys
import cv2
import numpy as np
import imutils
sys.path.append('/usr/local/python')

from openpose import pyopenpose as op


def scale_transform(coords):
    """
    Parameters:
    coords (25x3 ndarray): array of (x,y,c) coordinates

    Returns:
    ndarray: coords scaled to 1x1 with center at (0,0)=a
    ndarray: confidence scores of each joint
    """
    coords, scores = coords[:, :, :-1], coords[:, :, -1]
    diff = coords.max(axis=1) - coords.min(axis=1)
    diff_max = np.max(diff, axis=0)
    mean = coords.mean(axis=1).reshape(
                coords.shape[0],
                1,
                coords.shape[-1]
    )
    out = (coords - mean) / diff_max    # Add overlay to show results

    return out, scores


def make_vector(poseKeypoints):
    """
    Parameters:
    poseKeypoints (ndarray): Single person output from OpenPose

    Returns:
    ndarray: scaled, transformed, normalized row vector
    """
    N, D, C = poseKeypoints.shape
    coords, pose_scores = scale_transform(poseKeypoints)
    pose_scores = pose_scores.reshape((N, D, 1)) # 각 위치에 맞게 잘 디텍딩 됐는지? 그런느낌인가...?
    # print('pose_scores: {}'.format(pose_scores))
    coords_vec = np.concatenate([coords, pose_scores], axis=2)
    coords_vec /= np.linalg.norm(coords_vec, axis=2)[:, :, np.newaxis]

    return coords_vec


def get_ordinal_score(score):
    """
    Parameters:
    score (float): similarity score between two poses
                   between 0 and 1

    Returns:
    string: string text of the results
    float: transparency value
    tuple: color of overlay
    """
    score = score
    alpha = 0.2
    overlay_color = (0, 0, 0)
    if score > 0.90: # 여기에 cos 유사도 조건을 추가하는 조건도 있을듯!
        out = "Perfect!"
        overlay_color =  (205, 255, 15)
    elif score > 0.6:
        out = "Good!"
        overlay_color = (255, 150, 0)
    # elif score > 0.298:
    #     out = "Bad!"
    else:
        out = "Miss!"

    return score, out, alpha, overlay_color


def crop_image(full_image, w, h):
    full_image = cv2.resize(full_image,
                            (w, h))
    # full_image = imutils.resize(full_image, width=480)
    w_min = w // 2 - (w // 4)
    w_max = w // 2 + (w // 4)
    out = full_image[:h, w_min:w_max]
    return out


def crop_targetimage(full_image, w, h):
    full_image = cv2.resize(full_image,
                            (w, h))
    # full_image = imutils.resize(full_image, width=480)
    # w_min = w // 2 - (w // 4)
    # w_max = w // 2 + (w // 4)
    out = full_image[:h, :w]
    return out

# webcam 영상 받아와서 가로, 세로 사이즈 지정
# def get_webcam(w, h):
#     stream = cv2.VideoCapture(0)
#     if (stream.isOpened() is False):
#         print("Error opening video stream or file")
#         raise SystemExit(1)
#     stream.set(3, w)
#     stream.set(4, h)
#     return stream

def get_image(stream, w, h):
    ret, img_original = stream.read()
    # Reset video if reached end
    if not img_original.any():
        # 캡쳐한 영상의 속성 값을 설정, 원래 영상의 프레임 수로 지정
        stream.set(cv2.CAP_PROP_POS_FRAMES, 25)
        ret, img_original = stream.read()
    
    # # image 뒤집는 코드
    img = cv2.flip(
            crop_image(
                img_original,
                w, h
            ), 1)

    return img

def get_target_image(stream, w, h):
    ret, img_original = stream.read()
    # Reset video if reached end
    if not img_original.any():
        stream.set(cv2.CAP_PROP_POS_FRAMES, 25)
        ret, img_original = stream.read()
		
    img = crop_targetimage(
                img_original,
                w, h)
    return img

def label_img(opWrapper, img):
    datum = op.Datum()
    datum.cvInputData = img
    # opWrapper.waitAndEmplace(op.VectorDatum([datum]))
    # opWrapper.waitAndPop(op.VectorDatum([datum]))
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    return datum

# Display comment
def puttext(screen_out2, score1, score2):
    font = cv2.FONT_HERSHEY_COMPLEX 
    accuracy = str(int(score1*100))
    cv2.putText(screen_out2, accuracy, (553, 70), font, 0.7, (205, 255, 15), 2, cv2.LINE_AA)
    cv2.putText(screen_out2, 'ACCURACY', (520, 95), font, 0.5, (205, 255, 15), 1, cv2.LINE_AA)
    cv2.circle(screen_out2, (560,75), 45, (205, 255, 15), 2)
    cv2.putText(screen_out2, ' ' + score2, (520, 145), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)

    # Display comment
def puttext2(screen_out2, score1, score2):
    font = cv2.FONT_HERSHEY_DUPLEX 
    accuracy = str(int(score1*100))
    cv2.circle(screen_out2, (400,75), 45, (53, 53, 53), -1)
    cv2.putText(screen_out2, accuracy, (387, 70), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)
    cv2.putText(screen_out2, 'ACCURACY', (362, 95), font, 0.48, (205, 255, 15), 1, cv2.LINE_AA)
    if score2 == 'Perfect!':
        cv2.putText(screen_out2, ' ' + score2, (347, 145), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)

    else:
        cv2.putText(screen_out2, ' ' + score2, (355, 145), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)

def puttext3(screen_out2,  angle_R_arm,angle_L_arm,angle_R_leg,angle_L_leg,angle_R_len,angle_L_len):
    font = cv2.FONT_HERSHEY_DUPLEX 
    cv2.rectangle(screen_out2, (120,515), (370,480), (20, 20, 20), -1)
    # cv2.circle(screen_out2, (200,200), 45, (53, 53, 53), -1)
    cv2.putText(screen_out2, 'Down your arm' , (130, 500), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(screen_out2, str(int(angle_R_arm-90)), (265, 500), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(screen_out2, 'degrees', (295, 500), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.circle(screen_out2, (400,75), 45, (53, 53, 53), -1)
    # cv2.putText(screen_out2, accuracy, (387, 70), font, 0.7, (205, 255, 15), 2, cv2.LINE_AA)
    # cv2.putText(screen_out2, 'ACCURACY', (360, 95), font, 0.5, (205, 255, 15), 1, cv2.LINE_AA)
    # else:
    #     if angle_R_len >=130 and angle_L_len >=130:
    #         if angle_R_arm>=100: # 오른팔 내려 
    #             cv2.putText(screen_out2, 'Down your arm' , (20, 500), font, 0.5, (205, 255, 15), 1, cv2.LINE_AA)
    #             cv2.putText(screen_out2, str(int(angle_R_arm-90)), (250, 500), font, 0.6, (205, 255, 15), 1, cv2.LINE_AA)
    #             cv2.putText(screen_out2, 'degrees', (300, 500), font, 0.5, (205, 255, 15), 1, cv2.LINE_AA)
    #             flag=1
    # cv2.putText(screen_out2, '왼팔을', (20, 500), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)
    # cv2.putText(screen_out2, str(int(angle_L_arm-90)),'도 움직이세요', (20, 500), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)

    # cv2.putText(screen_out2, '오른팔을' + (angle_R_arm-90),'도 움직이세요', (20, 500), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)

    # if angle_R_arm>=100 and angle_L_arm>=100: # 오른팔 내리고 왼팔 내려  
    #     #  cv2.putText(screen_out2, ' ' + score2, (355, 145), font, 0.7, (205, 255, 15), 1, cv2.LINE_AA)
    # elif angle_R_arm>=100 and angle_L_arm<=80: # 오른팔 내리고 왼팔 올려
    # elif angle_R_arm>=100 and angle_L_arm>=100: # 오른팔 올리고 왼팔 올려
    # elif angle_R_arm>=100 and angle_L_arm>=100: # 오른팔 올리고 왼팔 내려
    # elif angle_R_arm>=100: # 오른팔 내려
    # elif angle_R_arm<=80: # 오른팔 올려
    # elif angle_L_arm>=100: # 왼팔 내려
    # elif angle_L_arm<=80: # 왼팔 올려
     
## util.py 에 추가한 코드
def scale_transform2(crds):
    crds, scores = crds[:,:-1], crds[:,:-1]
    diff = crds.max(axis=1) - crds.min(axis=1)
    diff_max = np.max(diff, axis=0)
    mean=crds.mean(axis=1)
    a=(crds[:,0]-mean) / diff_max
    b=(crds[:,1]-mean) / diff_max
    out=np.concatenate((a.reshape(-1,1),b.reshape(-1,1)),axis=0).reshape(1,-1)
#     out=np.concatenate((a.reshape(-1,1),b.reshape(-1,1)),axis=0).reshape(25,2)
    return out
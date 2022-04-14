import sys
import cv2
import os
import argparse
import numpy as np

sys.path.append('/usr/local/python')

from openpose import pyopenpose as op

# Flags
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='../openpose/models/')
#parser.add_argument('--image_dir', type=str, default='./data')
#parser.add_argument('--image_dir', type=str, default='/home/piai/CV-pose-detection/data/squat_cross')
parser.add_argument('--image_dir', type=str, default='/home/piai/CV-pose-detection/all_image_total_folder/all_image_total/data_total/twist_and_curlup_total')
#parser.add_argument('--output_dir', type=str, default='./coords')
parser.add_argument('--output_dir', type=str, default='/home/piai/CV-pose-detection/all_image_total_folder/all_image_total_coord')
parser.add_argument('--net_resolution', type=str, default='176x176')
parser.add_argument('--number_people_max', type=int, default=1)
args = parser.parse_args()

# Custom Params
params = dict()
params['model_folder'] = args.model_folder
params['net_resolution'] = args.net_resolution
params['number_people_max'] = args.number_people_max
params['display'] = 0
params['disable_multi_thread'] = True
# params["model_pose"] = "MPII"  // when we use COCO


# Make output dir
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Read filepaths for images
classes = {}
for f in os.scandir(args.image_dir):
    if f.is_dir():
        filenames = [g.path for g in os.scandir(f.path) if g.is_file() and g.path.endswith(('.png', '.jpg'))]
        classes[f.name] = filenames

# Start openpose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
#
count = 1
#
i = 0
# Get coordinates and scores for all images
for c in classes:
    i += 1

    # Prepare file name
    out_dir_name = os.path.join(args.output_dir, c)
    out_arr = []

    for f in classes[c]:
        image_name = os.path.basename(f)
        #print('2222',os.path.basename(f))
        # Label image
        print("Labeling image {}".format(f),count)
        datum = op.Datum()
        imageToProcess = cv2.imread(f)
        if imageToProcess is None:
            continue
        datum.cvInputData = imageToProcess
        #opWrapper.waitAndEmplace([datum])
        #opWrapper.waitAndPop([datum])
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        if type(datum.poseKeypoints) == np.ndarray and \
           datum.poseKeypoints.size > 0:
            out_arr.append(datum.poseKeypoints)
            cv2.imshow("", datum.cvOutputData)
            #cv2.waitKey(0)
            #cv2.waitKey(50)
            cv2.imwrite('/home/piai/CV-pose-detection/all_image_total_folder/all_image_total_skelation/11/n'+image_name, datum.cvOutputData)
            count += 1
    # Save file
    np.save(out_dir_name, out_arr)
    print("Done with {}/{}".format(i, len(c)))

print("Complete!")

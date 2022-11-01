import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

from keras.preprocessing.image import img_to_array
import imutils
import cv2
import time
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]
emotion_cal = [0,0,0,0,0,0,0]
graph_label = ["smile", "non-smile"]

#prev_time = 0

#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture('react-webcam-interview-test.mp4')
#fps = camera.get(cv2.CAP_PROP_FPS) #30frame -> 1초에 1프레임
fps=2 #1초에 15프레임

try:
    if not os.path.exists('./frames'[:-4]):
        os.makedirs('./frames'[:-4])
except OSError:
    print ('Error: Creating directory. ' +  './frames'[:-4])

count = 0

while(camera.isOpened()):
    ret, image = camera.read()

    if ret==False:
        break
    
    if(int(camera.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
        cv2.imwrite('./frames'[:-4] + "/frame%d.jpg" % count, image)
        print('Saved frame number :', str(int(camera.get(1))))
        count += 1      
camera.release()


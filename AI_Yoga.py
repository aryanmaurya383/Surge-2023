#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !pip install opencv-python mediapipe numpy


# In[3]:


# from google.colab.patches import cv2_imshow
# from google.colab import drive
# drive.mount("/content/gdrive")
import imutils
import requests
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# # Reference image
# # src="https://i.imgur.com/3j8BPdc.png"

#

# In[4]:


for lndmrk in mp_pose.PoseLandmark:
    print(lndmrk.value, lndmrk)


# In[5]:


def caluclate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1],  c[0]-b[0]) - \
        np.arctan2(a[1]-b[1],  a[0]-b[0])
    angle = np.abs(radians*180/np.pi)

    if(angle > 180):
        angle = 360-angle

    return angle


# In[6]:


# shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
# elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
# wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
# caluclate_angle(shoulder,elbow,wrist)


# In[7]:


# import requests
# import imutils
# # url = "http://172.17.57.233:8080/shot.jpg"
# # img_resp = requests.get(url)
# # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
# # img = cv2.imdecode(img_arr, -1)
# # img = imutils.resize(img, width=1000, height=1800)


# # Video Feed (press q to quit)
# # cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("v1.mp4")
# cap.set(cv2.CAP_PROP_FPS, 60)

# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     # while True:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         # img_resp = requests.get(url)
#         # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#         # img = cv2.imdecode(img_arr, -1)
#         # frame = imutils.resize(img, width=1000, height=1800)

#         # Recolor Image
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image = cv2.flip(image, 1)
#         # image = cv2.rotate(image, 2)
#         image = cv2.resize(
#             image, (int(image.shape[1]*0.4), int(image.shape[0]*0.4)))

#         image.flags.writeable = False
#         # Make Detection
#         results = pose.process(image)
#         # print(results)
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Extract landmarks
#         try:
#             landmarks = results.pose_landmarks.landmark

#             # Get Coordinates
# shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
# elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
# wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
# knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
#              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
# hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
#             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
# ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
#               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
# shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
# elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
#                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
# wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
# knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
#               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
# hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
#              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
# ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
#                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
# shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
# elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
#          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
# wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
#          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
# Get Angle
# left_shoulder_angle = int(caluclate_angle(
#     elbow_left, shoulder_left, hip_left))
# right_shoulder_angle = int(caluclate_angle(
#     elbow_right, shoulder_right, hip_right))
# left_elbow_angle = int(caluclate_angle(
#     wrist_left, elbow_left, shoulder_left))
# right_elbow_angle = int(caluclate_angle(
#     wrist_right, elbow_right, shoulder_right))
# left_hip_angle = int(caluclate_angle(
#     shoulder_left, hip_left, knee_left))
# right_hip_angle = int(caluclate_angle(
#     shoulder_right, hip_right, knee_right))
# left_knee_angle = int(caluclate_angle(
#     hip_left, knee_left, ankle_left))
# right_knee_angle = int(caluclate_angle(
#     hip_right, knee_right, ankle_right))
#             # shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#             #             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#             # elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
#             #          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#             # wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
#             #          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
#             # Get Angle
#             size_of_video_feed = [image.shape[1], image.shape[0]]
#             angle = int(caluclate_angle(shoulder, elbow, wrist))

#             # Visualize angle
#             cv2.putText(image,  str(angle), tuple(np.multiply(elbow, size_of_video_feed).astype(
#                 int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

#             # Display Frame rate
#             cv2.putText(image,  "FRAME "+str(int(cap.get(cv2.CAP_PROP_FPS))), (10, 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

#             cv2.circle(image, tuple(np.multiply([landmarks[15].x, landmarks[15].y],size_of_video_feed).astype(int)), 12, (0, 0, 0), 5)
#             # print(landmarks)
#         except:
#             pass

#         # Render the dedtection on original image
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(
#                                       color=(245, 117, 66), thickness=2, circle_radius=2),
#                                   mp_drawing.DrawingSpec(
#                                       color=(245, 66, 230), thickness=2, circle_radius=2)
#                                   )

#         cv2.imshow('Mediapipe', image)
#         print(landmarks[26])

#         if(cv2.waitKey(10) & 0xFF == ord('q')):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
# # img = cv.imread('/content/gdrive/MyDrive/AI Yoga/img1.jpg')
# # cv2_imshow(img)


# In[17]:

# url = "http://172.17.57.233:8080/shot.jpg"
# img_resp = requests.get(url)
# img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
# img = cv2.imdecode(img_arr, -1)
# img = imutils.resize(img, width=1000, height=1800)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    image = cv2.imread("./tst2.jpg")
    # image = cv2.imread("./tst3.png")
    # img_resp = requests.get(url)
    # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    # img = cv2.imdecode(img_arr, -1)
    # frame = imutils.resize(img, width=1000, height=1800)

    # Recolor Image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    # image = cv2.rotate(image, 2)
    # image = cv2.resize(
    #     image, (int(image.shape[1]*0.4), int(image.shape[0]*0.4)))

    image.flags.writeable = False
    # Make Detection
    results = pose.process(image)
    print(results)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        # Get Coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        # shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        #             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        # elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        #          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        # wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
        #          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        # Get Angle
        size_of_video_feed = [image.shape[1], image.shape[0]]
        angle = int(caluclate_angle(shoulder, elbow, wrist))

        # Visualize angle
        cv2.putText(image,  str(angle), tuple(np.multiply(elbow, size_of_video_feed).astype(
            int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        # Display Frame rate
        # cv2.putText(image,  "FRAME "+str(int(cap.get(cv2.CAP_PROP_FPS))), (10, 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.circle(image, tuple(np.multiply(
            [landmarks[15].x, landmarks[15].y], size_of_video_feed).astype(int)), 12, (0, 0, 0), 5)
        # print(landmarks)
    except:
        pass

    # Render the dedtection on original image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=2)
                              )

    cv2.imshow('Mediapipe', image)
    # print(landmarks[26])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(landmarks)
    print(type(landmarks))
    # cap.release()
    # img = cv.imread('/content/gdrive/MyDrive/AI Yoga/img1.jpg')
    # cv2_imshow(img)


# In[ ]:

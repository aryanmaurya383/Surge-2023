# %%
# !pip install opencv-python mediapipe numpy


# %%
# from google.colab.patches import cv2_imshow
# from google.colab import drive
# drive.mount("/content/gdrive")
import math
import requests
import cv2
import mediapipe as mp
import numpy as np
import imutils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# %% [markdown]
# # Reference image
# # src="https://i.imgur.com/3j8BPdc.png"

# %% [markdown]
#

# %%
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1],  c[0]-b[0]) - \
        np.arctan2(a[1]-b[1],  a[0]-b[0])
    angle = np.abs(radians*180/np.pi)

    if(angle > 180):
        angle = 360-angle

    return angle


def calculate_len(a, b):
    a = np.array(a)  # First point
    b = np.array(b)  # Last point
    r_sq = (a[0]-b[0])**2 + (a[1]-b[1])**2
    if(len(a) == 3):
        r_sq = r_sq+(a[2]-b[2])**2
    r = math.sqrt(r_sq)
    return r


# %%
def return_detect_landmarks(image, detection_confidence=0.5, tracking_confidence=0.5):
    with mp_pose.Pose(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence) as pose:
        # Recolor Image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.flip(image, 1)
        # image = cv2.rotate(image, 2)
        # image = cv2.resize(image, (int(image.shape[1]*0.4), int(image.shape[0]*0.4)))

        image.flags.writeable = False
        # Make Detection
        results = pose.process(image)
        image.flags.writeable = True

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(245, 255, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(
                                      color=(200, 166, 230), thickness=2, circle_radius=2)
                                  )
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hide_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 29, 30, 31, 32]
        for i in hide_landmarks:
            results.pose_landmarks.landmark[i].visibility = 0
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
        # for hiding landmarks not to show
        return results.pose_landmarks


# %%
image = cv2.imread("tst2.jpg")
lnd_tree_pose = return_detect_landmarks(image)

# %%
image = cv2.imread("./wrong_pose_tree.jpg")
detecting_pose = return_detect_landmarks(image)
detecting_pose_landmark = detecting_pose.landmark

# %%
# For IP webcam
# url = "http://172.17.57.233:8080/shot.jpg"
# img_resp = requests.get(url)
# img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
# img = cv2.imdecode(img_arr, -1)
# img = imutils.resize(img, width=1000, height=1800)


# %%

# import requests
# import imutils
# # url = "http://172.17.57.233:8080/shot.jpg"
# # img_resp = requests.get(url)
# # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
# # img = cv2.imdecode(img_arr, -1)
# # img = imutils.resize(img, width=1000, height=1800)
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

#     # image = cv2.imread("./tst2.jpg")
#     image = cv2.imread("./tst6.png")
#     # image = cv2.imread("./tst3.png")
#     # img_resp = requests.get(url)
#     # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#     # img = cv2.imdecode(img_arr, -1)
#     # frame = imutils.resize(img, width=1000, height=1800)

#     # Recolor Image
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # image = cv2.flip(image, 1)
#     # image = cv2.rotate(image, 2)
#     # image = cv2.resize(
#     #     image, (int(image.shape[1]*0.4), int(image.shape[0]*0.4)))

#     image.flags.writeable = False
#     # Make Detection
#     results = pose.process(image)
#     # print(results)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     # Extract landmarks
#     try:
#         landmarks = results.pose_landmarks.landmark

#         # Get Coordinates
#         shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#         elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#         wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

#         # Get Angle
#         size_of_video_feed = [image.shape[1], image.shape[0]]
#         angle = int(calculate_angle(shoulder, elbow, wrist))

#         # Visualize angle
#         cv2.putText(image,  str(angle), tuple(np.multiply(elbow, size_of_video_feed).astype(
#             int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

#         # Display Frame rate
#         # cv2.putText(image,  "FRAME "+str(int(cap.get(cv2.CAP_PROP_FPS))), (10, 20),
#         #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
#         # diff = [tree_pose[0][0]-landmarks[0].x,
#         #         tree_pose[0][1]-landmarks[0].y, tree_pose[0][2]-landmarks[0].z,  0]
#         # print(diff)
#         # print(tree_pose[0]-diff)
#         # for index in tree_pose:
#         #     temp=[index[0]-diff[0], index[1]-diff[1], index[2]-diff[2], index[3]-diff[3]]
#         #     cv2.circle(image, tuple(np.multiply(
#         #     [temp[0], temp[1]], size_of_video_feed).astype(int)), 12, (0, 0, 0), 5)

#         # cv2.circle(image, tuple(np.multiply(
#         #     [landmarks[15].x, landmarks[15].y], size_of_video_feed).astype(int)), 12, (0, 0, 0), 5)
#         # print(landmarks)
#     except:
#         pass

#     # print(type(landmarks[0]), type(tree_pose[0]))
#     # Render the dedtection on original image
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(
#                                       color=(245, 255, 66), thickness=2, circle_radius=2),
#                                   mp_drawing.DrawingSpec(
#                                       color=(200, 166, 230), thickness=2, circle_radius=2)
#                                   )

#     cv2.imshow("img", image)
#     cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     detecting_pose_landmark = results.pose_landmarks.landmark


# %%
# # import requests
# # import imutils
# # url = "http://172.23.146.190:8080/shot.jpg"
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
#             shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#             elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#             wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
#             knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
#             hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#             ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
#                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
#             shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#             elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
#                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#             wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
#             knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
#                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
#             hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
#                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#             ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
#                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
#             shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#             elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
#                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#             wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
#                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
#             # Get Angle
#             left_shoulder_angle = int(calculate_angle(
#                 elbow_left, shoulder_left, hip_left))
#             right_shoulder_angle = int(calculate_angle(
#                 elbow_right, shoulder_right, hip_right))
#             left_elbow_angle = int(calculate_angle(
#                 wrist_left, elbow_left, shoulder_left))
#             right_elbow_angle = int(calculate_angle(
#                 wrist_right, elbow_right, shoulder_right))
#             left_hip_angle = int(calculate_angle(
#                 shoulder_left, hip_left, knee_left))
#             right_hip_angle = int(calculate_angle(
#                 shoulder_right, hip_right, knee_right))
#             left_knee_angle = int(calculate_angle(
#                 hip_left, knee_left, ankle_left))
#             right_knee_angle = int(calculate_angle(
#                 hip_right, knee_right, ankle_right))

#             # Get Angle
#             size_of_video_feed = [image.shape[1], image.shape[0]]
#             angle = int(calculate_angle(shoulder, elbow, wrist))

#             # Visualize angle
#             cv2.putText(image,  str(angle), tuple(np.multiply(elbow, size_of_video_feed).astype(
#                 int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

#             # Display Frame rate
#             cv2.putText(image,  "FRAME "+str(int(cap.get(cv2.CAP_PROP_FPS))), (10, 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

#             # print(diff)
#             # print(tree_pose[0]-diff)
#             # ind=0
#             # for index in tree_pose:
#             #     diff = [tree_pose[ind][0]-landmarks[ind].x,
#             #         tree_pose[ind][1]-landmarks[ind].y, tree_pose[ind][2]-landmarks[ind].z,  0]
#             #     temp = [index[0]-diff[0], index[1]-diff[1],
#             #             index[2]-diff[2], index[3]-diff[3]]
#             #     ind=ind+1
#             #     cv2.circle(image, tuple(np.multiply(
#             #         [temp[0], temp[1]], size_of_video_feed).astype(int)), 12, (0, 0, 0), 5)

#                 # cv2.circle(image, tuple(np.multiply([landmarks[15].x, landmarks[15].y],size_of_video_feed).astype(int)), 12, (0, 0, 0), 5)
#                 # # print(landmarks)
#         except:
#             pass

#         # Render the dedtection on original image
#         mp_drawing.draw_landmarks(image, lnd_tree_pose, mp_pose.POSE_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(
#                                       color=(25, 25, 66), thickness=2, circle_radius=2),
#                                   mp_drawing.DrawingSpec(
#                                       color=(0, 255, 20), thickness=2, circle_radius=2)
#                                   )

#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(
#                                       color=(245, 117, 66), thickness=2, circle_radius=2),
#                                   mp_drawing.DrawingSpec(
#                                       color=(245, 66, 230), thickness=2, circle_radius=2)
#                                   )

#         cv2.imshow('Mediapipe', image)
#         # print(landmarks[26])

#         if(cv2.waitKey(10) & 0xFF == ord('q')):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
# # img = cv.imread('/content/gdrive/MyDrive/AI Yoga/img1.jpg')
# # cv2_imshow(img)


# %%
# left_shoulder = [lnd_tree_pose.landmark[11].x,
#                  lnd_tree_pose.landmark[11].y,
#                  lnd_tree_pose.landmark[11].z]
# left_hip = [lnd_tree_pose.landmark[23].x,
#             lnd_tree_pose.landmark[23].y,
#             lnd_tree_pose.landmark[23].z]
# right_shoulder = [lnd_tree_pose.landmark[12].x,
#                   lnd_tree_pose.landmark[12].y,
#                   lnd_tree_pose.landmark[12].z]
# right_hip = [lnd_tree_pose.landmark[24].x,
#              lnd_tree_pose.landmark[24].y,
#              lnd_tree_pose.landmark[24].z]
# left_elbow = [lnd_tree_pose.landmark[11].x,
#                  lnd_tree_pose.landmark[11].y,
#                  lnd_tree_pose.landmark[11].z]
# left_knee = [lnd_tree_pose.landmark[23].x,
#             lnd_tree_pose.landmark[23].y,
#             lnd_tree_pose.landmark[23].z]
# right_elbow = [lnd_tree_pose.landmark[12].x,
#                   lnd_tree_pose.landmark[12].y,
#                   lnd_tree_pose.landmark[12].z]
# right_knee = [lnd_tree_pose.landmark[24].x,
#              lnd_tree_pose.landmark[24].y,
#              lnd_tree_pose.landmark[24].z]

# len_perfect = (calculate_len(left_shoulder, left_hip) +
#                calculate_len(right_shoulder, right_hip))/2
# print(len_perfect)
# #-----------------------------------------------------------

# left_shoulder = [landmarks[11].x,
#                  landmarks[11].y,
#                  landmarks[11].z]
# left_hip = [landmarks[23].x,
#             landmarks[23].y,
#             landmarks[23].z]
# right_shoulder = [landmarks[12].x,
#                   landmarks[12].y,
#                   landmarks[12].z]
# right_hip = [landmarks[24].x,
#              landmarks[24].y,
#              landmarks[24].z]


# len_detected = (calculate_len(left_shoulder, left_hip) +
#                calculate_len(right_shoulder, right_hip))/2
# print(len_detected)


# %%
def transform_landmarks(perfect_pose, detecting_pose):
    perfect_pose_landmark = perfect_pose.landmark
    detecting_pose_landmark = detecting_pose.landmark
    copy_perfect_pose_landmark = perfect_pose_landmark

    # centroid = [(perfect_pose_landmark[11].x+perfect_pose_landmark[12].x+perfect_pose_landmark[23].x+perfect_pose_landmark[24].x)/4,
    #             (perfect_pose_landmark[11].y+perfect_pose_landmark[12].y+perfect_pose_landmark[23].y+perfect_pose_landmark[24].y)/4, (perfect_pose_landmark[11].z+perfect_pose_landmark[12].z+perfect_pose_landmark[23].z+perfect_pose_landmark[24].z)/4]

    # centroid2 = [(detecting_pose_landmark[11].x+detecting_pose_landmark[12].x+detecting_pose_landmark[23].x+detecting_pose_landmark[24].x)/4,
    #             (detecting_pose_landmark[11].y+detecting_pose_landmark[12].y+detecting_pose_landmark[23].y+detecting_pose_landmark[24].y)/4, (detecting_pose_landmark[11].z+detecting_pose_landmark[12].z+detecting_pose_landmark[23].z+detecting_pose_landmark[24].z)/4]

    # b = calculate_len([detecting_pose_landmark[11].x, detecting_pose_landmark[11].y, detecting_pose_landmark[11].z], [detecting_pose_landmark[12].x, detecting_pose_landmark[12].y,
    #                 detecting_pose_landmark[12].z])/calculate_len([copy_perfect_pose_landmark[11].x, copy_perfect_pose_landmark[11].y, copy_perfect_pose_landmark[11].z], [copy_perfect_pose_landmark[12].x, copy_perfect_pose_landmark[12].y, copy_perfect_pose_landmark[12].z])
    # r = calculate_len([detecting_pose_landmark[11].x, detecting_pose_landmark[11].y, detecting_pose_landmark[11].z], [detecting_pose_landmark[23].x, detecting_pose_landmark[23].y,
    #                 detecting_pose_landmark[23].z])/calculate_len([copy_perfect_pose_landmark[11].x, copy_perfect_pose_landmark[11].y, copy_perfect_pose_landmark[11].z], [copy_perfect_pose_landmark[23].x, copy_perfect_pose_landmark[23].y, copy_perfect_pose_landmark[23].z])

    b = calculate_len([detecting_pose_landmark[11].x, detecting_pose_landmark[11].y, detecting_pose_landmark[11].z], [detecting_pose_landmark[12].x, detecting_pose_landmark[11].y,
                      detecting_pose_landmark[12].z])/calculate_len([copy_perfect_pose_landmark[11].x, copy_perfect_pose_landmark[11].y, copy_perfect_pose_landmark[11].z], [copy_perfect_pose_landmark[12].x, copy_perfect_pose_landmark[12].y, copy_perfect_pose_landmark[12].z])
    # r = calculate_len([ detecting_pose_landmark[11].y, detecting_pose_landmark[11].z], [ detecting_pose_landmark[23].y,detecting_pose_landmark[23].z])/calculate_len([ copy_perfect_pose_landmark[11].y, copy_perfect_pose_landmark[11].z], [copy_perfect_pose_landmark[23].y, copy_perfect_pose_landmark[23].z])

    centroid = [(perfect_pose_landmark[11].x+perfect_pose_landmark[12].x)/2, (perfect_pose_landmark[11].y +
                                                                              perfect_pose_landmark[12].y)/2, (perfect_pose_landmark[11].z+perfect_pose_landmark[12].z)/2]

    # Setting and scaling X axis only
    r = 1
    # 11
    diff = [(detecting_pose_landmark[11].x-perfect_pose_landmark[11].x),
            (detecting_pose_landmark[11].y-perfect_pose_landmark[11].y)]
    # diff = [(perfect_pose_landmark[11].x-centroid[0])*(b-1), (perfect_pose_landmark[11].y-centroid[1])*(r-1)]
    ind = [11, 13, 15, 17, 19, 21]
    for i in ind:
        # print(i)
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]

    # 12
    diff = [(detecting_pose_landmark[12].x-perfect_pose_landmark[12].x),
            (detecting_pose_landmark[12].y-perfect_pose_landmark[12].y)]
    # diff = [(perfect_pose_landmark[12].x-centroid[0])*(b-1), (perfect_pose_landmark[12].y-centroid[1])*(r-1)]
    ind = [12, 14, 16, 18, 20, 22]
    for i in ind:
        # print(i)
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]

    b = calculate_len([detecting_pose_landmark[23].x, detecting_pose_landmark[23].y, detecting_pose_landmark[23].z], [detecting_pose_landmark[24].x, detecting_pose_landmark[24].y, detecting_pose_landmark[24].z])/calculate_len(
        [copy_perfect_pose_landmark[23].x, copy_perfect_pose_landmark[23].y, copy_perfect_pose_landmark[23].z], [copy_perfect_pose_landmark[24].x, copy_perfect_pose_landmark[24].y, copy_perfect_pose_landmark[24].z])

    centroid = [(perfect_pose_landmark[23].x+perfect_pose_landmark[24].x)/2, (perfect_pose_landmark[23].y +
                                                                              perfect_pose_landmark[24].y)/2, (perfect_pose_landmark[23].z+perfect_pose_landmark[24].z)/2]

    # 24
    diff = [(detecting_pose_landmark[24].x-perfect_pose_landmark[24].x),
            (detecting_pose_landmark[24].y-perfect_pose_landmark[24].y)]
    # diff = [(perfect_pose_landmark[24].x-centroid[0])*(b-1), (perfect_pose_landmark[24].y-centroid[1])*(r-1)]
    ind = [24, 26, 28, 30, 32]
    for i in ind:
        # print(i)
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]

    # 23
    diff = [(detecting_pose_landmark[23].x-perfect_pose_landmark[23].x),
            (detecting_pose_landmark[23].y-perfect_pose_landmark[23].y)]
    # diff = [(perfect_pose_landmark[23].x-centroid[0])*(b-1), (perfect_pose_landmark[23].y-centroid[1])*(r-1)]
    ind = [23, 25, 27, 29, 31]
    for i in ind:
        # print(i)
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]
    # ----------------------------------------------------------------------------------

    # #Setting and scaling Y axis only
    # b=1
    # left= calculate_len([ detecting_pose_landmark[11].y, detecting_pose_landmark[11].z], [ detecting_pose_landmark[23].y, detecting_pose_landmark[23].z]
    #                   )/calculate_len([ perfect_pose_landmark[11].y, perfect_pose_landmark[11].z], [perfect_pose_landmark[23].y, perfect_pose_landmark[23].z])

    # right = calculate_len([ detecting_pose_landmark[12].y, detecting_pose_landmark[12].z], [ detecting_pose_landmark[24].y, detecting_pose_landmark[24].z]
    #               )/calculate_len([ perfect_pose_landmark[12].y, perfect_pose_landmark[12].z], [ perfect_pose_landmark[24].y, perfect_pose_landmark[24].z])

    # r=(right+left)/2
    # # r=1
    # centroid = [(perfect_pose_landmark[23].x+perfect_pose_landmark[24].x)/2, perfect_pose_landmark[23].y, (perfect_pose_landmark[23].z+perfect_pose_landmark[24].z)/2]

    # # 11
    # diff = [(perfect_pose_landmark[11].x-centroid[0])*(b-1),
    #         (perfect_pose_landmark[11].y-centroid[1])*(r-1)]
    # ind = [11, 13, 15, 17, 19, 21]
    # for i in ind:
    #     # print(i)
    #     # perfect_pose_landmark[i].x += diff[0]
    #     perfect_pose_landmark[i].y += diff[1]

    # centroid = [(perfect_pose_landmark[23].x+perfect_pose_landmark[24].x)/2,
    #             perfect_pose_landmark[24].y, (perfect_pose_landmark[23].z+perfect_pose_landmark[24].z)/2]
    # # 12
    # diff = [(perfect_pose_landmark[12].x-centroid[0])*(b-1),
    #         (perfect_pose_landmark[12].y-centroid[1])*(r-1)]
    # ind = [12, 14, 16, 18, 20, 22]
    # for i in ind:
    #     # print(i)
    #     perfect_pose_landmark[i].y += diff[1]
    #     # perfect_pose_landmark[i].x += diff[0]

    # r = calculate_len([detecting_pose_landmark[12].x, detecting_pose_landmark[12].y, detecting_pose_landmark[12].z], [detecting_pose_landmark[24].x, detecting_pose_landmark[24].y, detecting_pose_landmark[24].z]
    #               )/calculate_len([copy_perfect_pose_landmark[12].x, copy_perfect_pose_landmark[12].y, copy_perfect_pose_landmark[12].z], [copy_perfect_pose_landmark[24].x, copy_perfect_pose_landmark[24].y, copy_perfect_pose_landmark[24].z])

    # centroid = [(perfect_pose_landmark[12].x+perfect_pose_landmark[24].x)/2, (perfect_pose_landmark[12].y +
    #                                                                           perfect_pose_landmark[24].y)/2, (perfect_pose_landmark[12].z+perfect_pose_landmark[24].z)/2]

    # # 12
    # diff = [(perfect_pose_landmark[12].x-centroid[0])*(b-1),
    #         (perfect_pose_landmark[12].y-centroid[1])*(r-1)]
    # ind = [12, 14, 16, 18, 20, 22]
    # for i in ind:
    #     # print(i)
    #     perfect_pose_landmark[i].x += diff[0]
    #     perfect_pose_landmark[i].y += diff[1]

    # # 24
    # diff = [(perfect_pose_landmark[24].x-centroid[0])*(b-1),
    #         (perfect_pose_landmark[24].y-centroid[1])*(r-1)]
    # ind = [24, 26, 28, 30, 32]
    # for i in ind:
    #     # print(i)
    #     perfect_pose_landmark[i].x += diff[0]
    #     perfect_pose_landmark[i].y += diff[1]
    # ----------------------------------------------------------------------------------
    # 13
    # tree_hgh = (np.abs(copy_perfect_pose_landmark[11].y-copy_perfect_pose_landmark[13].y) +
    #             np.abs(copy_perfect_pose_landmark[12].y-copy_perfect_pose_landmark[14].y))
    # detected_hgh = (np.abs(detecting_pose_landmark[11].y-detecting_pose_landmark[13].y) + np.abs(
    #     detecting_pose_landmark[12].y-detecting_pose_landmark[14].y))
    r = calculate_len([detecting_pose_landmark[11].x, detecting_pose_landmark[11].y], [detecting_pose_landmark[13].x, detecting_pose_landmark[13].y]) / \
        calculate_len([perfect_pose_landmark[11].x, perfect_pose_landmark[11].y], [
                      perfect_pose_landmark[13].x, perfect_pose_landmark[13].y])
    centroid = [perfect_pose_landmark[11].x, perfect_pose_landmark[11].y]
    diff = [(perfect_pose_landmark[13].x-centroid[0])*(r-1),
            (perfect_pose_landmark[13].y-centroid[1])*(r-1)]
    ind = [13, 15, 17, 19, 21]
    for i in ind:
        # print(i)
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]

    # 14
    r = calculate_len([detecting_pose_landmark[12].x, detecting_pose_landmark[12].y], [detecting_pose_landmark[14].x, detecting_pose_landmark[14].y]) / \
        calculate_len([perfect_pose_landmark[12].x, perfect_pose_landmark[12].y], [
                      perfect_pose_landmark[14].x, perfect_pose_landmark[14].y])

    centroid = [perfect_pose_landmark[12].x, perfect_pose_landmark[12].y]
    diff = [(perfect_pose_landmark[14].x-centroid[0])*(r-1),
            (perfect_pose_landmark[14].y-centroid[1])*(r-1)]
    ind = [14, 16, 18, 20, 22]
    for i in ind:
        # print(i)
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]

    # 15
    # tree_hgh = (np.abs(copy_perfect_pose_landmark[15].y-copy_perfect_pose_landmark[13].y) +
    #             np.abs(copy_perfect_pose_landmark[16].y-copy_perfect_pose_landmark[14].y))
    # detected_hgh = (np.abs(detecting_pose_landmark[15].y-detecting_pose_landmark[13].y) + np.abs(
    #     detecting_pose_landmark[16].y-detecting_pose_landmark[14].y))
    r = calculate_len([detecting_pose_landmark[15].x, detecting_pose_landmark[15].y], [detecting_pose_landmark[13].x, detecting_pose_landmark[13].y]) / \
        calculate_len([perfect_pose_landmark[15].x, perfect_pose_landmark[15].y], [
                      perfect_pose_landmark[13].x, perfect_pose_landmark[13].y])
    centroid = [perfect_pose_landmark[13].x, perfect_pose_landmark[13].y]
    diff = [(perfect_pose_landmark[15].x-centroid[0])*(r-1),
            (perfect_pose_landmark[15].y-centroid[1])*(r-1)]
    ind = [15, 17, 19, 21]
    for i in ind:
        # print(i)
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]

    # 16
    r = calculate_len([detecting_pose_landmark[14].x, detecting_pose_landmark[14].y], [detecting_pose_landmark[16].x, detecting_pose_landmark[16].y]) / \
        calculate_len([perfect_pose_landmark[14].x, perfect_pose_landmark[14].y], [
                      perfect_pose_landmark[16].x, perfect_pose_landmark[16].y])

    centroid = [perfect_pose_landmark[14].x, perfect_pose_landmark[14].y]
    diff = [(perfect_pose_landmark[16].x-centroid[0])*(r-1),
            (perfect_pose_landmark[16].y-centroid[1])*(r-1)]
    ind = [16, 18, 20, 22]
    for i in ind:
        # print(i)
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]

    # 17
    tree_hgh = (np.abs(copy_perfect_pose_landmark[15].y-copy_perfect_pose_landmark[17].y) +
                np.abs(copy_perfect_pose_landmark[16].y-copy_perfect_pose_landmark[18].y))
    detected_hgh = (np.abs(detecting_pose_landmark[15].y-detecting_pose_landmark[17].y) + np.abs(
        detecting_pose_landmark[16].y-detecting_pose_landmark[18].y))
    r = detected_hgh/tree_hgh
    centroid = [perfect_pose_landmark[15].x, perfect_pose_landmark[15].y]
    diff = [(perfect_pose_landmark[17].x-centroid[0])*(r-1),
            (perfect_pose_landmark[17].y-centroid[1])*(r-1)]
    perfect_pose_landmark[17].x += diff[0]
    perfect_pose_landmark[17].y += diff[1]

    # 18
    centroid = [perfect_pose_landmark[16].x, perfect_pose_landmark[16].y]
    diff = [(perfect_pose_landmark[18].x-centroid[0])*(r-1),
            (perfect_pose_landmark[18].y-centroid[1])*(r-1)]
    perfect_pose_landmark[18].x += diff[0]
    perfect_pose_landmark[18].y += diff[1]

    # 19
    tree_hgh = (np.abs(copy_perfect_pose_landmark[15].y-copy_perfect_pose_landmark[19].y) +
                np.abs(copy_perfect_pose_landmark[16].y-copy_perfect_pose_landmark[20].y))
    detected_hgh = (np.abs(detecting_pose_landmark[15].y-detecting_pose_landmark[19].y) + np.abs(
        detecting_pose_landmark[16].y-detecting_pose_landmark[20].y))
    r = detected_hgh/tree_hgh
    centroid = [perfect_pose_landmark[15].x, perfect_pose_landmark[15].y]
    diff = [(perfect_pose_landmark[19].x-centroid[0])*(r-1),
            (perfect_pose_landmark[19].y-centroid[1])*(r-1)]
    perfect_pose_landmark[19].x += diff[0]
    perfect_pose_landmark[19].y += diff[1]

    # 20
    centroid = [perfect_pose_landmark[16].x, perfect_pose_landmark[16].y]
    diff = [(perfect_pose_landmark[20].x-centroid[0])*(r-1),
            (perfect_pose_landmark[20].y-centroid[1])*(r-1)]
    perfect_pose_landmark[20].x += diff[0]
    perfect_pose_landmark[20].y += diff[1]

    # 21
    tree_hgh = (np.abs(copy_perfect_pose_landmark[15].y-copy_perfect_pose_landmark[21].y) +
                np.abs(copy_perfect_pose_landmark[16].y-copy_perfect_pose_landmark[22].y))
    detected_hgh = (np.abs(detecting_pose_landmark[15].y-detecting_pose_landmark[21].y) + np.abs(
        detecting_pose_landmark[16].y-detecting_pose_landmark[22].y))
    r = detected_hgh/tree_hgh
    centroid = [perfect_pose_landmark[15].x, perfect_pose_landmark[15].y]
    diff = [(perfect_pose_landmark[21].x-centroid[0])*(r-1),
            (perfect_pose_landmark[21].y-centroid[1])*(r-1)]
    perfect_pose_landmark[21].x += diff[0]
    perfect_pose_landmark[21].y += diff[1]

    # 22
    tree_hgh = (np.abs(copy_perfect_pose_landmark[15].y-copy_perfect_pose_landmark[21].y) +
                np.abs(copy_perfect_pose_landmark[16].y-copy_perfect_pose_landmark[22].y))
    detected_hgh = (np.abs(detecting_pose_landmark[15].y-detecting_pose_landmark[21].y) + np.abs(
        detecting_pose_landmark[16].y-detecting_pose_landmark[22].y))
    r = detected_hgh/tree_hgh
    centroid = [perfect_pose_landmark[16].x, perfect_pose_landmark[16].y]
    diff = [(perfect_pose_landmark[22].x-centroid[0])*(r-1),
            (perfect_pose_landmark[22].y-centroid[1])*(r-1)]
    perfect_pose_landmark[22].x += diff[0]
    perfect_pose_landmark[22].y += diff[1]

    # ------------------------------------------------------------------------------------
    # 26
    r = calculate_len([detecting_pose_landmark[26].x, detecting_pose_landmark[26].y], [detecting_pose_landmark[24].x, detecting_pose_landmark[24].y]) / \
        calculate_len([copy_perfect_pose_landmark[26].x, copy_perfect_pose_landmark[26].y], [
                      copy_perfect_pose_landmark[24].x, copy_perfect_pose_landmark[24].y])
    centroid = [perfect_pose_landmark[24].x, perfect_pose_landmark[24].y]
    diff = [(perfect_pose_landmark[26].x-centroid[0])*(r-1),
            (perfect_pose_landmark[26].y-centroid[1])*(r-1)]
    ind = [26, 28, 30, 32]
    for i in ind:
        # print(i)
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]

    # 25
    r = calculate_len([detecting_pose_landmark[25].x, detecting_pose_landmark[25].y], [detecting_pose_landmark[23].x, detecting_pose_landmark[23].y]) / \
        calculate_len([copy_perfect_pose_landmark[25].x, copy_perfect_pose_landmark[25].y], [
                      copy_perfect_pose_landmark[23].x, copy_perfect_pose_landmark[23].y])

    centroid = [perfect_pose_landmark[23].x, perfect_pose_landmark[23].y]
    diff = [(perfect_pose_landmark[25].x-centroid[0])*(r-1),
            (perfect_pose_landmark[25].y-centroid[1])*(r-1)]
    ind = [25, 27, 29, 31]
    for i in ind:
        # print(i)
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]

    # 28
    r = calculate_len([detecting_pose_landmark[26].x, detecting_pose_landmark[26].y], [detecting_pose_landmark[28].x, detecting_pose_landmark[28].y]) / \
        calculate_len([copy_perfect_pose_landmark[28].x, copy_perfect_pose_landmark[28].y], [
                      copy_perfect_pose_landmark[26].x, copy_perfect_pose_landmark[26].y])
    centroid = [perfect_pose_landmark[26].x, perfect_pose_landmark[26].y]
    diff = [(perfect_pose_landmark[28].x-centroid[0])*(r-1),
            (perfect_pose_landmark[28].y-centroid[1])*(r-1)]
    ind = [28, 30, 32]
    for i in ind:
        # print(i)
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]

    # 27
    r = calculate_len([detecting_pose_landmark[27].x, detecting_pose_landmark[27].y], [detecting_pose_landmark[25].x, detecting_pose_landmark[25].y]) / \
        calculate_len([copy_perfect_pose_landmark[27].x, copy_perfect_pose_landmark[27].y], [
                      copy_perfect_pose_landmark[25].x, copy_perfect_pose_landmark[25].y])
    centroid = [perfect_pose_landmark[25].x, perfect_pose_landmark[25].y]
    diff = [(perfect_pose_landmark[27].x-centroid[0])*(r-1),
            (perfect_pose_landmark[27].y-centroid[1])*(r-1)]
    ind = [27, 29, 31]
    for i in ind:
        # print(i)
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]

    # 30
    tree_hgh = (np.abs(copy_perfect_pose_landmark[28].y-copy_perfect_pose_landmark[30].y) +
                np.abs(copy_perfect_pose_landmark[27].y-copy_perfect_pose_landmark[29].y))
    detected_hgh = (np.abs(detecting_pose_landmark[28].y-detecting_pose_landmark[30].y) + np.abs(
        detecting_pose_landmark[27].y-detecting_pose_landmark[29].y))
    r = detected_hgh/tree_hgh
    # r = calculate_len([detecting_pose_landmark[30].x, detecting_pose_landmark[30].y, detecting_pose_landmark[30].z], [detecting_pose_landmark[28].x, detecting_pose_landmark[28].y, detecting_pose_landmark[28].z])/calculate_len([copy_perfect_pose_landmark[30].x, copy_perfect_pose_landmark[30].y, copy_perfect_pose_landmark[30].z], [copy_perfect_pose_landmark[28].x,
    #                                                                                                                                                                                                                                                                                                                                        copy_perfect_pose_landmark[28].y, copy_perfect_pose_landmark[28].z])

    centroid = [perfect_pose_landmark[28].x, perfect_pose_landmark[28].y]
    diff = [(perfect_pose_landmark[30].x-centroid[0])*(r-1),
            (perfect_pose_landmark[30].y-centroid[1])*(r-1)]
    perfect_pose_landmark[30].x += diff[0]
    perfect_pose_landmark[30].y += diff[1]

    # 29
    centroid = [perfect_pose_landmark[27].x, perfect_pose_landmark[27].y]
    diff = [(perfect_pose_landmark[29].x-centroid[0])*(r-1),
            (perfect_pose_landmark[29].y-centroid[1])*(r-1)]
    perfect_pose_landmark[29].x += diff[0]
    perfect_pose_landmark[29].y += diff[1]

    # 32
    # r = calculate_len([detecting_pose_landmark[32].x, detecting_pose_landmark[32].y, detecting_pose_landmark[32].z], [detecting_pose_landmark[28].x, detecting_pose_landmark[28].y, detecting_pose_landmark[28].z])/calculate_len([copy_perfect_pose_landmark[32].x, copy_perfect_pose_landmark[32].y, copy_perfect_pose_landmark[32].z], [copy_perfect_pose_landmark[28].x,
    r = detected_hgh/tree_hgh
    #    copy_perfect_pose_landmark[28].y, copy_perfect_pose_landmark[28].z])

    centroid = [perfect_pose_landmark[28].x, perfect_pose_landmark[28].y]
    diff = [(perfect_pose_landmark[32].x-centroid[0])*(r-1),
            (perfect_pose_landmark[32].y-centroid[1])*(r-1)]
    perfect_pose_landmark[32].x += diff[0]
    perfect_pose_landmark[32].y += diff[1]

    # 31
    centroid = [perfect_pose_landmark[27].x, perfect_pose_landmark[27].y]
    diff = [(perfect_pose_landmark[31].x-centroid[0])*(r-1),
            (perfect_pose_landmark[31].y-centroid[1])*(r-1)]
    perfect_pose_landmark[31].x += diff[0]
    perfect_pose_landmark[31].y += diff[1]

    # --------------------------------------------------------------------------------
    ts_centroid = [(perfect_pose_landmark[11].x+perfect_pose_landmark[12].x+perfect_pose_landmark[23].x+perfect_pose_landmark[24].x)/4,
                   (perfect_pose_landmark[11].y+perfect_pose_landmark[12].y+perfect_pose_landmark[23].y+perfect_pose_landmark[24].y)/4]
    lnd_centroid = [(detecting_pose_landmark[11].x+detecting_pose_landmark[12].x+detecting_pose_landmark[23].x+detecting_pose_landmark[24].x)/4,
                    (detecting_pose_landmark[11].y+detecting_pose_landmark[12].y+detecting_pose_landmark[23].y+detecting_pose_landmark[24].y)/4]
    diff = [lnd_centroid[0]-ts_centroid[0], lnd_centroid[1]-ts_centroid[1]]
    # diff=[detecting_pose_landmark[0].x-perfect_pose_landmark[0].x,detecting_pose_landmark[0].y-perfect_pose_landmark[0].y]
    for i in range(32):
        perfect_pose_landmark[i].x += diff[0]
        perfect_pose_landmark[i].y += diff[1]

    return perfect_pose


# %%
lnd_tree_pose = transform_landmarks(lnd_tree_pose, detecting_pose)
lnd_tree_pose_landmark = lnd_tree_pose.landmark

# %%


def landmark2list(landmark):
    converted = [landmark.x, landmark.y, landmark.z]
    return converted


# %%
def calculate_angles_landmark(detecting_pose_landmark):
    left_shoulder = landmark2list(detecting_pose_landmark[11])
    left_hip = landmark2list(detecting_pose_landmark[23])
    right_shoulder = landmark2list(detecting_pose_landmark[12])
    right_hip = landmark2list(detecting_pose_landmark[24])
    left_elbow = landmark2list(detecting_pose_landmark[13])
    left_knee = landmark2list(detecting_pose_landmark[25])
    right_elbow = landmark2list(detecting_pose_landmark[14])
    right_knee = landmark2list(detecting_pose_landmark[26])
    left_wrist = landmark2list(detecting_pose_landmark[15])
    left_ankle = landmark2list(detecting_pose_landmark[27])
    right_wrist = landmark2list(detecting_pose_landmark[16])
    right_ankle = landmark2list(detecting_pose_landmark[28])

    left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
    right_shoulder_angle = calculate_angle(
        right_hip, right_shoulder, right_elbow)
    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = calculate_angle(
        right_shoulder, right_elbow, right_wrist)
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, left_ankle)

    display_angles = [[left_shoulder_angle, 11], [right_shoulder_angle, 12], [left_elbow_angle, 13], [right_elbow_angle, 14],
                      [left_hip_angle, 23], [right_hip_angle, 24], [left_knee_angle, 25], [right_knee_angle, 26]]
    return display_angles


# %%
display_angles = calculate_angles_landmark(detecting_pose_landmark)
perfect_angles = calculate_angles_landmark(lnd_tree_pose_landmark)


# %%
# import requests
# import imutils
# # url = "http://172.17.57.233:8080/shot.jpg"
# # img_resp = requests.get(url)
# # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
# # img = cv2.imdecode(img_arr, -1)
# # img = imutils.resize(img, width=1000, height=1800)
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

#     # image1 = cv2.imread("./scaledup.png")
#     # image = cv2.imread("./tst2.jpg")
#     # image1 = cv2.imread("./tst2.jpg")
#     image = cv2.imread("./wrong_pose_tree.jpg")
#     image1 = cv2.imread("./wrong_pose_tree.jpg")
#     # image = cv2.imread("./tst3.png")
#     # img_resp = requests.get(url)
#     # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#     # img = cv2.imdecode(img_arr, -1)
#     # frame = imutils.resize(img, width=1000, height=1800)
#     # cap=cv2.VideoCapture(0)
#     flag=1
#     # while cap.isOpened():
#     # Recolor Image
#         # ret, frame = cap.read()
#     # image = imutils.resize(image, width=450, height=720)

#     size_of_video_feed = [image.shape[1], image.shape[0]]
#     if flag:
#         image1 = cv2.resize(image1, (size_of_video_feed[0], size_of_video_feed[1]))
#         image1 = cv2.flip(image1, 1)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.flip(image, 1)
#     # image = cv2.rotate(image, 2)
#     # image = cv2.resize(
#     #     image, (int(image.shape[1]*0.4), int(image1.shape[0]*0.4)))
#     # image1 = cv2.resize(
#     #     image1, (int(image1.shape[1]*0.4), int(image1.shape[0]*0.4)))
#     # image.flags.writeable = False
#     # Make Detection
#     results = pose.process(image)
#     # print(results)
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#     # Extract landmarks
#     try:
#         hide_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 29, 30, 31, 32]
#         for i in hide_landmarks:
#             results.pose_landmarks.landmark[i].visibility = 0
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                 mp_drawing.DrawingSpec(
#                                     color=(245, 255, 66), thickness=2, circle_radius=2),
#                                 mp_drawing.DrawingSpec(
#                                     color=(200, 166, 230), thickness=2, circle_radius=2)
#                                 )
#         # mp_drawing.draw_landmarks(image, lnd_tree_pose, mp_pose.POSE_CONNECTIONS,
#         #                             mp_drawing.DrawingSpec(
#         #                                 color=(25, 25, 66), thickness=2, circle_radius=2),
#         #                             mp_drawing.DrawingSpec(
#         #                                 color=(0, 255, 20), thickness=2, circle_radius=2)
#         #                             )
#         #visualize angle
#         for i in display_angles:
#             cv2.putText(image,  str(int(i[0])), tuple(np.multiply([results.pose_landmarks.landmark[i[1]].x, results.pose_landmarks.landmark[i[1]].y], size_of_video_feed).astype(
#                 int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

#     except:
#         pass
#     if flag:
#         mp_drawing.draw_landmarks(image1, lnd_tree_pose, mp_pose.POSE_CONNECTIONS,
#                                 mp_drawing.DrawingSpec(
#                                     color=(25, 25, 66), thickness=2, circle_radius=2),
#                                 mp_drawing.DrawingSpec(
#                                     color=(0, 255, 20), thickness=2, circle_radius=2)
#                                 )
#         for i in perfect_angles:
#             cv2.putText(image1,  str(int(i[0])), tuple(np.multiply([lnd_tree_pose_landmark[i[1]].x, lnd_tree_pose_landmark[i[1]].y], size_of_video_feed).astype(
#                 int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
#     # Hori = np.concatenate((image, image1), axis=1)

#     # cv2.imshow("img", Hori)
#     if flag:
#         cv2.imshow("img1", image1)
#     flag=0
#     cv2.imshow("img", image)
#     cv2.waitKey(0)
#     # if(cv2.waitKey(10) & 0xFF == ord('q')):
#     #     break
#     # cap.release()
#     cv2.destroyAllWindows()


# %%
def get_detected_image(perfect_pose_landmarks, detected_image):

    try:
        size_of_video_feed = [detected_image.shape[1], detected_image.shape[0]]
        detecting_pose = return_detect_landmarks(detected_image)
        detecting_pose_landmark = detecting_pose.landmark

        perfect_pose_landmarks = transform_landmarks(
            perfect_pose_landmarks, detecting_pose)
        perfect_pose_landmark = perfect_pose_landmarks.landmark

        display_angles = calculate_angles_landmark(detecting_pose_landmark)
        perfect_angles = calculate_angles_landmark(perfect_pose_landmark)

        error_in_angles = np.absolute(
            np.array(display_angles)-np.array(perfect_angles))
        error_in_angles = np.divide(error_in_angles, perfect_angles)
        score = 1-np.sum(error_in_angles)/len(error_in_angles)
        # print("YYES")
        # score=1
        score *= 100
        print(score)
        mp_drawing.draw_landmarks(detected_image, detecting_pose, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(245, 255, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(
                                      color=(200, 166, 230), thickness=2, circle_radius=2)
                                  )
        mp_drawing.draw_landmarks(detected_image, perfect_pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(25, 25, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(
                                      color=(0, 255, 20), thickness=2, circle_radius=2)
                                  )

        cv2.putText(detected_image,  "SCORE : "+str(int(score)), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 0), 2, cv2.LINE_AA)
    except:
        pass
    return [detected_image, score]


# %%
# #Remember to comment the waitkey lines form return_detect_landmarks function before running video

# import requests
# import imutils
# # url = "http://172.17.57.233:8080/shot.jpg"
# # img_resp = requests.get(url)
# # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
# # img = cv2.imdecode(img_arr, -1)
# # img = imutils.resize(img, width=1000, height=1800)
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

#     # image1 = cv2.imread("./scaledup.png")
#     # image = cv2.imread("./tst2.jpg")
#     # image1 = cv2.imread("./tst2.jpg")
#     # image = cv2.imread("./tst6.png")
#     # image1 = cv2.imread("./tst6.png")
#     # image = cv2.imread("./tst3.png")
#     # img_resp = requests.get(url)
#     # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#     # img = cv2.imdecode(img_arr, -1)
#     # frame = imutils.resize(img, width=1000, height=1800)
#     image1 = cv2.imread("tst2.jpg")
#     lnd_tree_pose = return_detect_landmarks(image)


#     # cap = cv2.VideoCapture(0)
#     cap = cv2.VideoCapture("v1.mp4")
#     flag = 0
#     while cap.isOpened():
#     # Recolor Image
#         ret, frame = cap.read()
#         image = imutils.resize(frame, width=420, height=820)

#         size_of_video_feed = [image.shape[1], image.shape[0]]
#         if flag:
#             image1 = cv2.resize(image1, (size_of_video_feed[0], size_of_video_feed[1]))
#             image1 = cv2.flip(image1, 1)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = cv2.flip(image, 1)
# #-----------------------------------------------------------------
#         try:
#             detecting_pose = return_detect_landmarks(image)
#             detecting_pose_landmark = detecting_pose.landmark

#             lnd_tree_pose = transform_landmarks(lnd_tree_pose, detecting_pose)
#             lnd_tree_pose_landmark = lnd_tree_pose.landmark

#             display_angles = calculate_angles_landmark(detecting_pose_landmark)
#             perfect_angles = calculate_angles_landmark(lnd_tree_pose_landmark)
#         except:
#             pass
# #----------------------------------------------------------------------
#         # image = cv2.rotate(image, 2)
#         # image = cv2.resize(
#         #     image, (int(image.shape[1]*0.4), int(image1.shape[0]*0.4)))
#         # image1 = cv2.resize(
#         #     image1, (int(image1.shape[1]*0.4), int(image1.shape[0]*0.4)))
#         # image.flags.writeable = False
#         # Make Detection
#         results = pose.process(image)
#         # print(results)
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Extract landmarks
#         try:
#             mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                     mp_drawing.DrawingSpec(
#                                         color=(245, 255, 66), thickness=2, circle_radius=2),
#                                     mp_drawing.DrawingSpec(
#                                         color=(200, 166, 230), thickness=2, circle_radius=2)
#                                     )
#             mp_drawing.draw_landmarks(image, lnd_tree_pose, mp_pose.POSE_CONNECTIONS,
#                                       mp_drawing.DrawingSpec(
#                                           color=(25, 25, 66), thickness=2, circle_radius=2),
#                                       mp_drawing.DrawingSpec(
#                                           color=(0, 255, 20), thickness=2, circle_radius=2)
#                                       )
#             #visualize angle
#             for i in display_angles:
#                 cv2.putText(image,  str(int(i[0])), tuple(np.multiply([results.pose_landmarks.landmark[i[1]].x, results.pose_landmarks.landmark[i[1]].y], size_of_video_feed).astype(
#                     int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

#         except:
#             pass
#         if flag:
#             mp_drawing.draw_landmarks(image1, lnd_tree_pose, mp_pose.POSE_CONNECTIONS,
#                                     mp_drawing.DrawingSpec(
#                                         color=(25, 25, 66), thickness=2, circle_radius=2),
#                                     mp_drawing.DrawingSpec(
#                                         color=(0, 255, 20), thickness=2, circle_radius=2)
#                                     )
#             for i in perfect_angles:
#                 cv2.putText(image1,  str(int(i[0])), tuple(np.multiply([lnd_tree_pose_landmark[i[1]].x, lnd_tree_pose_landmark[i[1]].y], size_of_video_feed).astype(
#                     int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
#         # Hori = np.concatenate((image, image1), axis=1)

#         # cv2.imshow("img", Hori)
#         if flag:
#             cv2.imshow("img1", image1)
#         flag=0
#         cv2.imshow("img", image)
#         if(cv2.waitKey(10) & 0xFF == ord('q')):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

# %%
image = cv2.imread("./tst2.jpg")
img1 = cv2.imread("./tst6.png")
lnd_tree_pose = return_detect_landmarks(image)
detected_img, score = get_detected_image(lnd_tree_pose, img1)
cv2.imshow("img", detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

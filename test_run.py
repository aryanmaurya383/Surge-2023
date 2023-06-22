import detector_file as detector
import cv2
image = cv2.imread("./tst2.jpg")
img1 = cv2.imread("./tst6.png")
lnd_tree_pose = detector.return_detect_landmarks(image)
detected_img, score = detector.get_detected_image(lnd_tree_pose, img1)
cv2.imshow("img", detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

from io import BytesIO
import cv2
from flask import Flask, request
from PIL import Image
import base64
import numpy as np
import detector_file as detector
app = Flask(__name__)

@app.route('/')
def does_it_work():
    return 'It works!'

@app.route('/upload', methods=['POST'])
def upload_image():
    base64_image = request.form.get('image')
    image_bytes = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_bytes))

    # Convert the PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image1 = cv2.imread("./tst2.jpg")
    # Process the image using the provided functions
    lnd_tree_pose = detector.return_detect_landmarks(image1)
    detected_img, score = detector.get_detected_image(lnd_tree_pose, img_cv)
    # cv2.imshow("img", detected_img)
    # Convert the processed image back to PIL format
    processed_image = Image.fromarray(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))

    # Convert the processed image to base64
    buffered = BytesIO()
    processed_image.save(buffered, format="PNG")
    processed_image_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Return the processed image as a response
    return processed_image_base64

if __name__ == '__main__':
    app.run()

    
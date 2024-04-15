from asyncio import sleep
import cv2
from deepface.detectors import FaceDetector
import requests
from tensorflow.keras.models import load_model
import os

def build_model():

    url = "https://drive.google.com/uc?export=download&id=1Tng7GuiGTj_nzDCnhoFAo7aLLf9TtEsf"

    model_name = "lean-mtcnn.keras"

    model_path = os.path.join(os.getcwd(), model_name)
    print(f"model path: {model_path}")

    if not os.path.exists(model_name):
        print(f"{model_name} not found, downloading from {url}")
        
        # Make a GET request to download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Save the file to the current directory
        with open(model_name, 'wb') as f:
            f.write(response.content)
        

        print(f"Downloaded {model_name} successfully.")


    else:
        print(f"{model_name} already exists, no download needed.")

    print("sleeping")
    while 1:
        sleep(5)

    print(f"loading {model_name}")

    face_detector = load_model('lean-mtcnn.keras')
    
    return face_detector


def detect_face(face_detector, img, align=True):

    resp = []

    detected_face = None
    img_region = [0, 0, img.shape[1], img.shape[0]]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mtcnn expects RGB but OpenCV read BGR
    detections = face_detector.detect_faces(img_rgb)

    if len(detections) > 0:

        for detection in detections:
            x, y, w, h = detection["box"]
            detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
            img_region = [x, y, w, h]
            confidence = detection["confidence"]

            if align:
                keypoints = detection["keypoints"]
                left_eye = keypoints["left_eye"]
                right_eye = keypoints["right_eye"]
                detected_face = FaceDetector.alignment_procedure(detected_face, left_eye, right_eye)

            resp.append((detected_face, img_region, confidence))

    return resp

if __name__ == "__main__":

    build_model()

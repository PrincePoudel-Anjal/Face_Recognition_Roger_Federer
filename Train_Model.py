import numpy as np
import pandas as pd
import os
# import tensorflow as tf
# import matplotlib.pyplot as plt
from keras_facenet import FaceNet
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
embedder = FaceNet()
def embeddings(image):
    img = image.astype('float32')
    img = np.expand_dims(img,axis = 0)
    ebdings = embedder.embeddings(img)
    if len(ebdings) == 0:
        return 0
    return ebdings[0]



def detect_and_crop(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))


    results = mp_face_detection.process(image)
    if not results.detections:
        return None

    bbox = results.detections[0].location_data.relative_bounding_box

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    width = int(bbox.width * w)     #width of the cropped image
    height = int(bbox.height * h)   #height of the cropped image
    x_start, y_start = int(bbox.xmin * w), int(bbox.ymin * h)
    cropped_image = image[y_start:y_start+height,x_start:x_start+width]
    print("Image is detected and Cropped Successfully")
    return cropped_image,x_start, y_start,height,width

def draw_rectangle(image,x_start,y_start,height,width):
    height_orig,width_orig,_ = image.shape
    scale_height = height_orig/224
    scale_width = width_orig/224
    y_start = int(y_start*scale_height)
    x_start = int(x_start*scale_width)
    height = int(height*scale_height)
    width = int(width*scale_width)

    cv2.rectangle(image, (x_start, y_start), (x_start + width, y_start + height), (0, 255, 0), 2)
    return image,x_start,y_start,width,height



known_embeddings = []
known_names = []
known = "Known"

def train_model():
    for face in os.listdir(known):
        path = os.path.join(known, face)
        for image in os.listdir(path):
            img = os.path.join(path, image)
            img = cv2.imread(img)
            output = detect_and_crop(img)
            if output is None:
                continue
            cropped_image, x_start, y_start, height, width = output
            known_embeddings.append(embeddings(cropped_image))
            known_names.append(face)

    print("Model is Trained")

train_model()
np.save("known_embeddings.npy", known_embeddings)
np.save("known_names.npy", known_names)














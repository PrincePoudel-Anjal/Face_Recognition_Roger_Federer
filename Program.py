import numpy as np
import os
from keras_facenet import FaceNet
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import cv2
import mediapipe as mp

known_embeddings = np.load("known_embeddings.npy")
known_names      = np.load("known_names.npy")


mp_face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)
embedder = FaceNet()
def embeddings(image):
    image = cv2.resize(image, (224,224))
    img = image.astype('float32')
    img = np.expand_dims(img,axis = 0)
    ebdings = embedder.embeddings(img)
    if len(ebdings) == 0:
        return 0
    return ebdings[0]



def detect_and_crop(image):   #It crops the face and gives location on Real UnKnownImage
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    return cropped_image,x_start, y_start,height,width

def draw_rectangle(image,x_start,y_start,height,width):
    cv2.rectangle(image, (x_start, y_start), (x_start + width, y_start + height), (0, 255, 0), 8)
    return image,x_start,y_start,height,width

# Face_Recognition:
best_match = None

video_path = os.path.join(r"Roger_Federer_Walk.mp4")
video = cv2.VideoCapture(video_path)

no_of_frames = 0     #no of frames
unknownembedding = [0]


while True:
    best_distance = 999
    ret,unknownimage = video.read()




    if no_of_frames % 1 == 0:
        # Detecting Faces
        detection_output = detect_and_crop(unknownimage)
        if detection_output is not None:
            cropped_unknownimage, x_start, y_start, height, width = detection_output
            unknownembedding = embeddings(cropped_unknownimage)
        else:

            no_of_frames = no_of_frames + 1

# I find embeddings on unknownimage i.e doing so I won't resize unknownimaze for imshow
            img = unknownimage.copy()
            img = cv2.resize(unknownimage, (500,600))

            cv2.imshow("face", img)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        # Comparing the detected face with Known_Faces
    for index, i in enumerate(known_embeddings):
        dist = np.linalg.norm(unknownembedding - i)
        if dist < best_distance:
            best_match = known_embeddings[index]
            best_distance = dist
        if dist < 0.9:
            unknownimage,x_start,y_start,height,width = draw_rectangle(unknownimage, x_start, y_start, height, width)
            cv2.putText(unknownimage, known_names[index], (x_start, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
    no_of_frames = no_of_frames + 1

    # I find embeddings on unknownimage i.e doing so I won't resize unknownimaze for imshow
    img = unknownimage.copy()
    img = cv2.resize(unknownimage, (500,600))

    cv2.imshow("face", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cv2.destroyAllWindows()



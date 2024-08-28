import cv2
import os
import numpy as np
from keras.models import load_model
from pygame import mixer

def initialize():
    mixer.init()
    sound = mixer.Sound('alarm.wav')

   
    face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
    left_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

  
    model = load_model('models/cnncat2.h5')

    return sound, face_cascade, left_eye_cascade, right_eye_cascade, model

def preprocess_eye(eye_image):
 
    eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    eye_image = cv2.resize(eye_image, (24, 24)) / 255.0
    eye_image = eye_image.reshape(24, 24, -1)
    eye_image = np.expand_dims(eye_image, axis=0)
    return eye_image

def detect_eye_state(eye_image, model):
   
    preprocessed_eye = preprocess_eye(eye_image)
    prediction = model.predict(preprocessed_eye)
    return np.argmax(prediction, axis=1)[0]

def main():
    sound, face_cascade, left_eye_cascade, right_eye_cascade, model = initialize()
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    s = 0
    thick = 2
    rpred = [99]
    lpred = [99]
    path = os.getcwd()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eyes = left_eye_cascade.detectMultiScale(gray)
        right_eyes = right_eye_cascade.detectMultiScale(gray)

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

     
        for (x, y, w, h) in right_eyes:
            right_eye = frame[y:y + h, x:x + w]
            rpred[0] = detect_eye_state(right_eye, model)
            break

        for (x, y, w, h) in left_eyes:
            left_eye = frame[y:y + h, x:x + w]
            lpred[0] = detect_eye_state(left_eye, model)
            break

        if rpred[0] == 0 and lpred[0] == 0:
            s += 1
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            s = 0
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f's: {s}', (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        if s > 10:
            cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
            sound.play()
            thick = thick + 2 if thick < 16 else thick - 2
            thick = max(thick, 2)
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thick)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()

import cv2
import numpy as np
from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#img = cv2.imread('group.jpg')
cap = cv2.VideoCapture(0)

while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #transform image to gray scale
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

    cv2.imshow('img',img)


    model = model_from_json(open("facial_expression_model_structure.json", "r").read())
    model.load_weights('facial_expression_model_weights.h5')  # load weights

    img_pixels = image.img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis=0)

    img_pixels /= 255

    predictions = model.predict(img_pixels)

    # find max indexed array
    max_index = np.argmax(predictions[0])

    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    emotion = emotions[max_index]

    cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if cv2.waitKey(1) &  0xFF == ord('q'):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
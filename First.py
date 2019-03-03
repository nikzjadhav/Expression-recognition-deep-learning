
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#img = cv2.imread('group.jpg')
cap = cv2.VideoCapture(0)

while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #transform image to gray scale
    faces = face_cascade.detectMultiScale(gray, 2, 5)
    for (x,y,w,h) in faces:
    #print(faces)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('img',img)
    if cv2.waitKey(1) &  0xFF == ord('q'):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
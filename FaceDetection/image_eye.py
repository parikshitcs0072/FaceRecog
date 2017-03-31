import numpy as np
import cv2
import dlib

PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('/home/parikshit/Pictures/ABC/jim99.sad')
img1 = cv2.imread('/home/parikshit/Pictures/jim199.pp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)


def get_landmarks(im):
    rects = detector(im, 1)
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    landmarks=get_landmarks(roi_color)
    left_eye=landmarks[40]
    right_eye=landmarks[45]
    #print right_eye[0, 0]
    cv2.imwrite('/home/parikshit/Desktop/abc.jpg',roi_color)
    eyes = eye_cascade.detectMultiScale(roi_gray,3)
    for (ex,ey,ew,eh) in eyes:
         print ex
         print ey
         if(ex>left_eye[0,0] and ew<left_eye[0,1]):
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
         if(ex>right_eye[0,0]and ew<right_eye[0,1]):
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

print landmarks[40]
print landmarks[45]
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()







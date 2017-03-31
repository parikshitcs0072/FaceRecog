import dlib
import cv2
import numpy

PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects)==0:
        return im
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im,landmarks):
    im=im.copy()
    for idx,point in enumerate(landmarks):
        # print idx
        # print point
        pos=(point[0,0],point[0,1])
        cv2.putText(im,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0,0,255))
        cv2.circle(im,pos,3,color=(0,255,255))
    return im

#image=cv2.imread('/home/parikshit/Desktop/abc.jpg')
#image=cv2.resize(image, (0,0), fx=0.5, fy=0.5)
# landmarks=get_landmarks(image)
#print landmarks
# img_l=annotate_landmarks(image,landmarks)
# cv2.imshow('im',img_l)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
i=0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        landmarks = get_landmarks(roi_color)
        img = annotate_landmarks(frame, landmarks)
        cv2.imwrite('/home/parikshit/Desktop/directory/img{}.jpg'.format(i), img)
        i=i+1
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
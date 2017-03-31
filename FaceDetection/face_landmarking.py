import dlib
import cv2
import numpy

PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(im):
    rects = detector(im, 1)
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im,landmarks):
    im=im.copy()
    for idx,point in enumerate(landmarks):
        pos=(point[0,0],point[0,1])
        cv2.putText(im,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0,0,255))
        cv2.circle(im,pos,3,color=(0,255,255))
    return im

image=cv2.imread('/home/parikshit/Pictures/ABC/jim99.sad')
#image=cv2.resize(image, (0,0), fx=0.5, fy=0.5)
landmarks=get_landmarks(image)
print landmarks
img_l=annotate_landmarks(image,landmarks)
cv2.imshow('im',img_l)
cv2.waitKey(0)
cv2.destroyAllWindows()
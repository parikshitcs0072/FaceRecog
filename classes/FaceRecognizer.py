import cv2
import numpy as np

class FaceRecognizer():
   def _init_(self,cascadePath):
       self.faceCascade= cv2.CascadeClassifier(cascadePath)
       self.recognizer = cv2.face.createLBPHFaceRecognizer()

   def trainRecognizer(self,images,labels):
       self.recognizer.train(images,np.array(labels))

   def updateRecognizer(self,images,labels):
       self.recognizer.update(images,np.array(labels))



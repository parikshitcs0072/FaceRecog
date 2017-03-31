import cv2, os
import numpy as np
from PIL import Image

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.face.createLBPHFaceRecognizer()

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    images = []
    labels = []
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    return images, labels


def get_images_and_labels1(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) ]
    images = []
    labels = []
    for image_path in image_paths:
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("jim", ""))
        faces = faceCascade.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    return images, labels

path = '../yalefaces'
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

recognizer.train(images, np.array(labels))


path1 = '/home/parikshit/Pictures/ABC'
images1, labels1 = get_images_and_labels1(path1)
recognizer.update(images1, np.array(labels1))
cv2.destroyAllWindows()

predict_image_pil = Image.open('/home/parikshit/Pictures/jim199.pp').convert('L')
predict_image = np.array(predict_image_pil, 'uint8')
faces = faceCascade.detectMultiScale(predict_image)

for (x, y, w, h) in faces:
    nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
    nbr_actual = int(os.path.split('/home/parikshit/Pictures/jim199.pp')[1].split(".")[0].replace("jim", ""))
    print "{} is Recognized as {} with confidence {}".format(nbr_actual,nbr_predicted,conf)
    cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
    cv2.waitKey(1000)
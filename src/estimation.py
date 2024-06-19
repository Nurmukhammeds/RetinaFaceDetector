import math
import cv2
import numpy as np

import warnings 

warnings.filterwarnings('ignore')

# make face detection on RetinaFace 
caffemodel = "resources/face_detector/deploy.prototxt"
deploy = "resources/face_detector/Widerface-RetinaFace.caffemodel"
detector = cv2.dnn.readNetFromCaffe(caffemodel, deploy)
detector_confidence = 0.5
print('RetinaFace Detector is loaded!')

def get_bbox_conf(img):
    height, width = img.shape[0], img.shape[1]
    aspect_ratio = width / height
    if img.shape[1] * img.shape[0] >= 192 * 192:
        img = cv2.resize(img,
                            (int(192 * math.sqrt(aspect_ratio)),
                            int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

    blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
    detector.setInput(blob, 'data')
    out = detector.forward('detection_out').squeeze()
    max_conf_index = np.argmax(out[:, 2])
    max_confidence = out[max_conf_index, 2]
    left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
    bbox = [int(left), int(top), int(right+1), int(bottom+1)]
    return bbox, max_confidence

# crop image via bounding boxes
def image_cropper(img, x, y, w, h):
    return img[y:h, x:w, :]

def calculate_area(x, y, w, h):
    return (w-x)*(h-y)
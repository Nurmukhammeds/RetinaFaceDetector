import cv2

from src.estimation import get_bbox_conf, image_cropper, calculate_area

import warnings
warnings.filterwarnings('ignore')


def face_detection(img_path):
    image = cv2.imread(img_path)
    bbox, max_confidence = get_bbox_conf(image)
    x, y, w, h = bbox
    updated_bbox = [abs(coordinate) for coordinate in [x, y, w, h]]
    x, y, w, h = updated_bbox
    face_area = calculate_area(x, y, w, h)

    face = image_cropper(image, x, y, w, h)

    cv2.imwrite('cropped_face.jpg', face)
    print('Found face.')
    
    return x, y, w, h, max_confidence, face_area 


if __name__=='__main__':
    face_detection('test_img.jpg')
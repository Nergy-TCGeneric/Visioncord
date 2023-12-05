import PIL.Image
import numpy as np
import cv2

from camera_util import calculate_angle
from dataset import load_cocos_class_names
from monodepth2 import DepthEstimator
from monodepth2.util import estimate_distance_from_disp, save_disparity_map_to_image
from yolo import *

def print_angles(box: BoundingBox, class_names: list[str]) -> None:
    x1, y1 = box.p_min
    x2, y2 = box.p_max

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    angle_x, angle_y = calculate_angle(test_img.size, (center_x, center_y))
    class_name = class_names[box.class_index]
    print(f"{class_name} : {angle_x} x {angle_y} degrees")

def print_distances(box: BoundingBox, dist: np.ndarray, class_names: list[str]) -> None:
    x1, y1 = box.p_min
    x2, y2 = box.p_max

    dist = estimated_distance[x1:x2, y1:y2]
    average = np.mean(dist)
    class_name = class_names[box.class_index]

    print(f"{class_name}, {average} mm")

def visualize_boxes(box: BoundingBox, img_array: np.ndarray) -> np.ndarray:
    x1, y1 = box.p_min
    x2, y2 = box.p_max
    img_array = cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0))
    return img_array

print("Loading YOLOv4-tiny detector..")
image_detector = YOLODetector(cfg_path='yolov4-tiny.cfg', weight_path='yolov4-tiny.weights')
print("Loading monodepth2 estimator..")
depth_estimator = DepthEstimator(encoder_weight_path='./models/encoder.pth', decoder_weight_path='./models/depth.pth')

# TODO: Should migrate to other dataset, since COCO dataset lacks of indoor things like door.. etc.
class_names = load_cocos_class_names('coco.names')

# Image to test with.
test_img: Image = PIL.Image.open('dog.jpg')
img_array = np.asarray(test_img)

# TODO: Running model doesn't have to be in sequential way. Maybe using a multiprocessing would be a key?
boxes = image_detector.predict(test_img)
disparities = depth_estimator.predict(test_img)
estimated_distance = estimate_distance_from_disp(disparities)

# Some applications
for box in boxes:
    print_angles(box, class_names)
    print_distances(box, estimated_distance, class_names)
    img_array = visualize_boxes(box, img_array)

save_disparity_map_to_image(disparities, 'disparities')

# Transform image to monodepth2 input format.
img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
cv2.imshow('rectangles', img_array)
cv2.waitKey(0)
cv2.destroyAllWindows()
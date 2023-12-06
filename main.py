import PIL.Image
import numpy as np
import cv2

from multiprocessing import Process
from multiprocessing.connection import Pipe

from camera_util import calculate_angle
from dataset import load_cocos_class_names
from monodepth2 import DepthEstimator
from monodepth2.util import estimate_distance_from_disp, save_disparity_map_to_image
from yolo import *

def print_angles(box: BoundingBox, image: Image, class_names: "list[str]") -> None:
    x1, y1 = box.p_min
    x2, y2 = box.p_max

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    angle_x, angle_y = calculate_angle(image.size, (center_x, center_y))
    class_name = class_names[box.class_index]
    print(f"{class_name} : {angle_x} x {angle_y} degrees")

def print_distances(box: BoundingBox, dist: np.ndarray, class_names: "list[str]") -> None:
    x1, y1 = box.p_min
    x2, y2 = box.p_max

    dist_region = dist[x1:x2, y1:y2]
    average = np.mean(dist_region)
    class_name = class_names[box.class_index]

    print(f"{class_name}, {average} mm")

def visualize_boxes(box: BoundingBox, img_array: np.ndarray) -> np.ndarray:
    x1, y1 = box.p_min
    x2, y2 = box.p_max
    img_array = cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0))
    return img_array


def main():
    print("Loading YOLOv4-tiny detector..")
    image_detector = YOLODetector(cfg_path='yolov4-tiny.cfg', weight_path='yolov4-tiny.weights')
    print("Loading monodepth2 estimator..")
    depth_estimator = DepthEstimator(encoder_weight_path='./models/encoder.pth', decoder_weight_path='./models/depth.pth')

    # TODO: Should migrate to other dataset, since COCO dataset lacks of indoor things like door.. etc.
    class_names = load_cocos_class_names('coco.names')

    # Image to test with.
    test_img: Image = PIL.Image.open('output.jpg')
    img_array = np.asarray(test_img)

    # Start child processes respectively.
    main_to_yolo_pipe, yolo_to_main_pipe = Pipe()
    yolo_process = Process(target=image_detector.predict, args=(yolo_to_main_pipe, ))
    yolo_process.start()

    main_to_depth_pipe, depth_to_main_pipe = Pipe()
    depth_process = Process(target=depth_estimator.predict, args=(depth_to_main_pipe, ))
    depth_process.start()

    print("Starting main loop..")

    while True:
        main_to_yolo_pipe.send(test_img)
        main_to_depth_pipe.send(test_img)

        boxes: "list[BoundingBox]" = main_to_yolo_pipe.recv()
        disparities: np.ndarray = main_to_depth_pipe.recv()
        estimated_distance = estimate_distance_from_disp(disparities)

        for box in boxes:
            print_angles(box, test_img, class_names)
            print_distances(box, estimated_distance, class_names)
            img_array = visualize_boxes(box, img_array)

        save_disparity_map_to_image(disparities, 'disparities')

        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imshow('rectangles', img_array)
        if cv2.waitKey(1) == ord('q'):
            break
    
    print("Q key pressed. terminating subprocesses..")
    # Since process are run in infinite loop, we should forcefully terminate it.
    yolo_process.terminate()
    depth_process.terminate()

    main_to_yolo_pipe.close()
    yolo_to_main_pipe.close() 
    main_to_depth_pipe.close()
    depth_to_main_pipe.close()

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
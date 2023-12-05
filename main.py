# TODO: Should remove this toolset by constructing model from scratch(!)
from tool.darknet2pytorch import Darknet
from torchvision.transforms import transforms
import torch
import numpy as np
import PIL.Image
import cv2

# RPi4 only.
import io
from picamera2 import Picamera2

import monodepth2.resnet_encoder
import monodepth2.depth_decoder
import matplotlib
import matplotlib.cm as cm

OBJECT_CONFIDENCE_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.5

# Known values of RPi camera v2.
HORIZONTAL_FOV = 62.2
VERTICAL_FOV = 48.8
ACTIVE_H_PIXELS_MAX = 3280
ACTIVE_V_PIXELS_MAX = 2464
FOCAL_LENGTH = 3.04 # in mm

# Experimental values for estimating depth from image.
BASELINE = 2250 # in mm

# Set up raspberry pi's camera
rpi_cam = Picamera2()
rpi_cam.start()
data = io.BytesIO()

def check_box_exceeds_nms_threshold(boxes: np.ndarray, confidences: np.ndarray):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1) * (y2 - y1)
    order = confidences.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.maximum(x2[idx_self], x2[idx_other])
        yy2 = np.maximum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        intersection = w * h
        iou = intersection / (area[order[0]] + area[order[1:]] - intersection)
        invalid = np.where(iou < NMS_IOU_THRESHOLD)[0]
        order = order[invalid + 1]

    return np.array(keep)

def non_max_suppression(prediction: torch.Tensor):
    box_arrays = prediction[0]
    confidences = prediction[1]
    num_of_classes = confidences.shape[2]

    # Presumably author just wanted to transfer data from other H/W accelerator to CPU
    # As this is mostly done in CPU.
    if type(box_arrays) != np.ndarray:
        box_arrays: np.ndarray = box_arrays.cpu().detach().numpy()
        confidences: np.ndarray = confidences.cpu().detach().numpy()

    # [batch, num, 1, 4] -> [batch, num, 4]
    box_arrays = box_arrays[:, :, 0]
    # [batch, num, num_classes] -> [batch, num]
    max_confidence: np.ndarray = np.max(confidences, axis=2)
    max_index: np.ndarray = np.argmax(confidences, axis=2)

    bboxes_batch = []

    for batch in range(box_arrays.shape[0]):
        batch_argwhere = max_confidence[batch] > OBJECT_CONFIDENCE_THRESHOLD
        l_box_array = box_arrays[batch, batch_argwhere, :]
        l_max_conf = max_confidence[batch, batch_argwhere]
        l_max_index = max_index[batch, batch_argwhere]

        bboxes = []
        # NMS for each classes.
        for yolo_class in range(num_of_classes):
            class_argwhere = l_max_index == yolo_class
            ll_box_array = l_box_array[class_argwhere, :] 
            ll_max_conf = l_max_conf[class_argwhere]
            ll_max_id = l_max_index[class_argwhere]

            keep = check_box_exceeds_nms_threshold(ll_box_array, ll_max_conf)

            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    # Truncate the decimal part. we don't need it in pixel system.
                    x1, x2 = int(img_width * ll_box_array[k, 0]), int(img_width * ll_box_array[k, 2])
                    y1, y2 = int(img_height * ll_box_array[k, 1]), int(img_height * ll_box_array[k, 3])
                    bboxes.append([x1, y1, x2, y2, ll_max_conf[k], ll_max_id[k]])
        
        bboxes_batch.append(bboxes)
    return bboxes_batch

def load_cocos_class_names(coco_file_name: str) -> "list[str]":
    class_names = []
    with open(coco_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)

    return class_names

def calculate_angle(image_size: "tuple[int, int]", point: "tuple[int, int]") -> "tuple[float, float]":
    # Simple calculation based on center point is a point with zero degree angle
    center_x, center_y = image_size[0] // 2, image_size[1] // 2
    angle_x = (point[0] - center_x) / image_size[0] * HORIZONTAL_FOV
    angle_y = (point[1] - center_y) / image_size[1] * VERTICAL_FOV
    return (angle_x, angle_y)

# https://github.com/nianticlabs/monodepth2/blob/b676244e5a1ca55564eb5d16ab521a48f823af31/layers.py#L16
def estimate_distance_from_disp(disp: np.ndarray, min_depth: float, max_disp: float) -> np.ndarray:
    # CAVEAT: monodepth2 is trained using KITTI dataset to yield consistent results.
    # Indeed, the disp is not metric but a relative one.
    min_disp = 1 / max_disp
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp * FOCAL_LENGTH * BASELINE
    return depth

# This program will be run on RPi4, so it should use CPU.
rpi_device = torch.device('cpu')

# YOLOv4-tiny configuration.
yolov4_tiny = Darknet('yolov4-tiny.cfg', inference=True).to(rpi_device)
yolov4_tiny.load_weights('yolov4-tiny.weights')
yolov4_tiny.eval()

# TODO: Should migrate to other dataset, since COCO dataset lacks of indoor things like door.. etc.
class_names = load_cocos_class_names('coco.names')

# YOLOv4-tiny only accepts image with 416x416px, resizing is necessary
yolo_img_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])

# Monodepth2 configuration. Assumes monocular 640x192 model.
# Configuring encoder.
monodepth_encoder = monodepth2.resnet_encoder.ResnetEncoder(18, False)
loaded_dict_enc = torch.load('./models/encoder.pth', map_location=rpi_device)

feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in monodepth_encoder.state_dict()}

monodepth_encoder.load_state_dict(filtered_dict_enc)
monodepth_encoder.to(rpi_device)
monodepth_encoder.eval()

# Configuring decoder.
monodepth_decoder = monodepth2.depth_decoder.DepthDecoder(
    num_ch_enc=monodepth_encoder.num_ch_enc, scales=range(4))
loaded_dict = torch.load('./models/depth.pth', map_location=rpi_device)
monodepth_decoder.load_state_dict(loaded_dict)

monodepth_decoder.to(rpi_device)
monodepth_decoder.eval()

# Resizing image.
monodepth_img_transform = transforms.Compose([
    transforms.Resize((feed_width, feed_height)),
    transforms.ToTensor()
])

# TODO: Running model doesn't have to be in sequential way. Maybe using a multiprocessing would be a key?
while True:
    test_img = rpi_cam.capture_image().convert("RGB")
    img_array = np.asarray(test_img)
    img_width, img_height = test_img.size

    # Transform image to yolo input format.
    yolo_transformed: torch.Tensor = yolo_img_transform(test_img).to(rpi_device)
    yolo_transformed = yolo_transformed.unsqueeze(0)

    boxes = []

    # Model inference : YOLOv4-tiny
    with torch.no_grad():
        output = yolov4_tiny(yolo_transformed)
        boxes = non_max_suppression(output)
        for box in boxes[0]:
            x1, y1, x2, y2, class_id = box[0], box[1], box[2], box[3], box[5]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            angle_x, angle_y = calculate_angle(test_img.size, (center_x, center_y))
            class_name = class_names[class_id]
            print(f"{class_name} : {angle_x} x {angle_y} degrees")
            img_array = cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0))

    # Transform image to monodepth2 input format.
    monodepth_transformed: torch.Tensor = monodepth_img_transform(test_img).to(rpi_device)
    monodepth_transformed = monodepth_transformed.unsqueeze(0)

    # Model inference : Monodepth2
    with torch.no_grad():
        features = monodepth_encoder(monodepth_transformed)
        output = monodepth_decoder(features)

        disp = output[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (img_height, img_width), mode="bilinear", align_corners=False)
        
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        # These values are default values from monodepth2 repository
        estimated_distance = estimate_distance_from_disp(disp_resized_np, 0.1, 100)
        for box in boxes[0]:
            x1, y1, x2, y2, name_idx = box[0], box[1], box[2], box[3], box[5]
            dist = estimated_distance[x1:x2, y1:y2]
            average = np.mean(dist)
            class_name = class_names[name_idx]
            print(f"{class_name}, {average} mm")

        vmax = np.percentile(disp_resized_np, 95)
        normalizer = matplotlib.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = PIL.Image.fromarray(colormapped_im)
        im.save('test_disp.jpg')

    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imshow('rectangles', img_array)
    cv2.waitKey(1)

cv2.destroyAllWindows()
rpi_cam.close()

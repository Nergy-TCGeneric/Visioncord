# TODO: Should remove this toolset by constructing model from scratch(!)
from tool.darknet2pytorch import Darknet
from torchvision.transforms import transforms
import torch
import numpy as np
from PIL import Image

OBJECT_CONFIDENCE_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.5

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
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], \
                                   ll_max_conf[k], ll_max_id[k]])
        
        bboxes_batch.append(bboxes)
    return bboxes_batch

def load_cocos_class_names(coco_file_name: str) -> list[str]:
    class_names = []
    with open(coco_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            class_names.append(line)

    return class_names

# This program will be run on RPi4, so it should use CPU.
rpi_device = torch.device('cpu')

model = Darknet('yolov4-tiny.cfg', inference=True).to(rpi_device)
model.print_network()
model.load_weights('yolov4-tiny.weights')
model.eval()

# TODO: Should migrate to other dataset, since COCO dataset lacks of indoor things like door.. etc.
class_names = load_cocos_class_names('coco.names')

# YOLOv4-tiny only accepts image with 416x416px, resizing is necessary
test_img = Image.open('dog.jpg')
image_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])
transformed: torch.Tensor = image_transform(test_img).to(rpi_device)
if transformed.ndimension() == 3:
    transformed = transformed.unsqueeze(0)

with torch.no_grad():
    output = model(transformed)
    boxes = non_max_suppression(output)
    for box in boxes[0]:
        print(class_names[box[5]], box[4])
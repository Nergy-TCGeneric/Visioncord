
# TODO: Should remove this toolset by constructing model from scratch(!)
from tool.darknet2pytorch import Darknet
from torchvision import transforms
import numpy as np
import torch
from PIL.Image import Image

from dataclasses import dataclass

OBJECT_CONFIDENCE_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.5

@dataclass
class Point:
    x: int
    y: int

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __iter__(self):
        yield self.x
        yield self.y

@dataclass
class BoundingBox:
    p_min: Point
    p_max: Point
    confidence: float
    class_index: int      

    def __init__(self, p_min: Point, p_max: Point, confidence: float, class_index: int):
        self.p_min = p_min
        self.p_max = p_max
        self.confidence = confidence
        self.class_index = class_index

class YOLODetector:
    __model: torch.nn.Module

    def __init__(self, cfg_path: str, weight_path: str):
        # Also, it only does inference here.
        self.__model = Darknet(cfg_path, inference=True)
        self.__model.load_weights(weight_path)
        self.__model.eval()

    def predict(self, image: Image) -> list[BoundingBox]:
        preprocessed = self.__preprocess_image(image)
        with torch.no_grad():
            output = self.__model(preprocessed)
            boxes = self.__non_max_suppression(output)
            return self.__postprocess_bboxes(boxes, image.size)

    def __preprocess_image(self, image: Image) -> torch.Tensor:
        # It uses CPU by default, because this program is supposed to run on RPi4.
        cpu_device = torch.device('cpu')

        # Image pre-processing
        image_transformer = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor()
        ])
        transformed: torch.Tensor = image_transformer(image).to(cpu_device)
        transformed = transformed.unsqueeze(0)

        return transformed
    
    def __postprocess_bboxes(self, bboxes: list, image_size: tuple[int, int]) -> list[BoundingBox]:
        processed: list[BoundingBox] = []
        image_width, image_height = image_size

        for raw_bbox in bboxes[0]:
            x1: int = int(raw_bbox[0] * image_width)
            y1: int = int(raw_bbox[1] * image_height)
            p_min = Point(x1, y1)

            x2: int = int(raw_bbox[2] * image_width)
            y2: int = int(raw_bbox[3] * image_height)
            p_max = Point(x2, y2)

            confidence: float = raw_bbox[4]
            class_id: int = raw_bbox[5]

            bbox = BoundingBox(p_min, p_max, confidence, class_id)
            processed.append(bbox)

        return processed

    def __extract_most_prominent_bboxes(self, boxes: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        # These are normalized values, as configured in model.
        x1: float = boxes[:, 0]
        y1: float = boxes[:, 1]
        x2: float = boxes[:, 2]
        y2: float = boxes[:, 3]

        area = (x2 - x1) * (y2 - y1)
        order = confidences.argsort()[::-1]

        # Check IOU to whether discard it or not
        keep_list = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]

            keep_list.append(idx_self)

            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.maximum(x2[idx_self], x2[idx_other])
            yy2 = np.maximum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)

            intersection = w * h
            iou = intersection / (area[order[0]] + area[order[1:]] - intersection)
            survived = np.where(iou < NMS_IOU_THRESHOLD)[0]
            order = order[survived + 1]

        return np.array(keep_list)

    def __non_max_suppression(self, prediction: torch.Tensor) -> list:
        box_arrays = prediction[0]
        confidences = prediction[1]
        num_of_classes = confidences.shape[2]

        # Presumably author just wanted to transfer data from other H/W accelerator to CPU
        # As this is mostly done in CPU.
        if type(box_arrays) != np.ndarray:
            box_arrays: np.ndarray = box_arrays.cpu().detach().numpy()
            confidences: np.ndarray = confidences.cpu().detach().numpy()

        # We discard the useless third index here.
        # [batch, num, 1, 4] -> [batch, num, 4]
        box_arrays = box_arrays[:, :, 0]

        # Get the most 'probable' one among confidences list.
        # [batch, num, num_classes] -> [batch, num]
        max_confidence: np.ndarray = np.max(confidences, axis=2)
        max_index: np.ndarray = np.argmax(confidences, axis=2)

        bboxes_batch: list = []
        for batch in range(box_arrays.shape[0]):
            # Filter out objects below threshold.
            survived_object_indices: np.ndarray = max_confidence[batch] > OBJECT_CONFIDENCE_THRESHOLD
            batch_box_array: np.ndarray = box_arrays[batch, survived_object_indices, :]
            batch_max_conf: np.ndarray = max_confidence[batch, survived_object_indices]
            batch_max_index: np.ndarray = max_index[batch, survived_object_indices]

            # NMS for each classes.
            bboxes: list[float, float, float, float, int, int] = []
            for cls in range(num_of_classes):
                selected_class_pred_indices: np.ndarray = batch_max_index == cls
                class_box_array: np.ndarray = batch_box_array[selected_class_pred_indices, :] 
                class_max_conf: np.ndarray = batch_max_conf[selected_class_pred_indices]
                class_max_index: np.ndarray = batch_max_index[selected_class_pred_indices]

                prominent = self.__extract_most_prominent_bboxes(class_box_array, class_max_conf)

                # If there're most prominent bboxes of a class, store it
                if prominent.size > 0:
                    selected_box_array: np.ndarray = class_box_array[prominent, :]
                    selected_box_confs: np.ndarray = class_max_conf[prominent]
                    selected_box_class_ids: np.ndarray = class_max_index[prominent]

                    for k in range(selected_box_array.shape[0]):
                        x1, x2 = selected_box_array[k, 0], selected_box_array[k, 2]
                        y1, y2 = selected_box_array[k, 1], selected_box_array[k, 3]
                        conf = selected_box_confs[k]
                        class_id = selected_box_class_ids[k]

                        bboxes.append([x1, y1, x2, y2, conf, class_id])            

            bboxes_batch.append(bboxes)

        return bboxes_batch
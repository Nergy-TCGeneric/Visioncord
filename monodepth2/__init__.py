from monodepth2.resnet_encoder import ResnetEncoder
from monodepth2.depth_decoder import DepthDecoder

from PIL.Image import Image, LANCZOS
import torch
from torchvision import transforms
import numpy as np

from multiprocessing.connection import PipeConnection

class DepthEstimator():
    __encoder: ResnetEncoder
    __encoder_shape: "tuple[int, int]"
    __decoder: DepthDecoder

    def __init__(self, encoder_weight_path: str, decoder_weight_path: str) -> None:
        default_device = torch.device('cpu')
        self.__setup_encoder(encoder_weight_path, default_device)
        self.__setup_decoder(decoder_weight_path, default_device)

    def __setup_encoder(self, encoder_weight_path: str, device: torch.device) -> None:
        # Default settings from niantics' repository
        self.__encoder = ResnetEncoder(18, False)
        loaded_dict_enc: dict = torch.load(encoder_weight_path, map_location=device)
        filtered_encoder_dict = {k: v for k, v in loaded_dict_enc.items() if k in self.__encoder.state_dict()}

        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        self.__encoder_shape = (feed_width, feed_height)

        self.__encoder.load_state_dict(filtered_encoder_dict)
        self.__encoder.to(device).eval()

    def __setup_decoder(self, decoder_weight_path: str, device: torch.device) -> None:
        # Default settings from niantics' repository
        self.__decoder = DepthDecoder(
            num_ch_enc=self.__encoder.num_ch_enc, scales=range(4))
        decoder_dict = torch.load(decoder_weight_path, map_location=device)
        self.__decoder.load_state_dict(decoder_dict)
        self.__decoder.to(device).eval()

    def predict(self, data_out: PipeConnection) -> None:
        while True:
            image: Image = data_out.recv()
            preprocessed = self.__preprocess_image(image)

            with torch.no_grad():
                features: torch.Tensor = self.__encoder(preprocessed)
                output: torch.Tensor = self.__decoder(features)
                disparity: torch.Tensor = output[("disp", 0)] 
                resized_disparity: torch.Tensor = torch.nn.functional.interpolate(
                    disparity, image.size, mode="bilinear", align_corners=False)
                resized_disparity = resized_disparity.squeeze().cpu().numpy()
                data_out.send(resized_disparity)

    def __preprocess_image(self, image: Image) -> torch.Tensor:
        # Unlike YOLO, this requires different resizing method.
        resized = image.resize(self.__encoder_shape, LANCZOS)
        transformed = transforms.ToTensor()(resized).unsqueeze(0)
        return transformed
 
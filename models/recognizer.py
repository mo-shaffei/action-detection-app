from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import numpy as np
import skvideo.io
import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ShortSideScale,
    UniformTemporalSubsample,
)
from pytorchvideo.models.hub.vision_transformers import mvit_base_32x3
from pytorchvideo.models.hub.slowfast import slowfast_r50


class RecognizerModel:
    def __init__(self, model_name: str = 'slowfast', person_bbox_threshold: float = 0.2, device: str = 'cpu'):
        if "slowfast" in model_name:
            self._model = slowfast_r50(pretrained=True)
            self._create_slowfast_transform()
        elif "mvit" in model_name:
            self._model = mvit_base_32x3(pretrained=True)
            self._create_mvit_transform()
        else:
            raise Exception(f'Invalid model name {model_name}')
        self._load_label_map()
        self._device = device
        self._person_bbox_threshold = person_bbox_threshold
        self._model_name = model_name
        self._load_detectron2()
        self._post_act = torch.nn.Softmax(dim=1)
        self._video_data = None
        self._preds = None
        self._clip_no = -1

    def _load_label_map(self, path: str = "models/kinetics_classnames.json"):
        with open(path, "r") as f:
            kinetics_classnames = json.load(f)

        # Create an id to label name mapping
        kinetics_id_to_classname = {}
        for k, v in kinetics_classnames.items():
            kinetics_id_to_classname[v] = str(k).replace('"', "")
        self._kinetics_id_to_classname = kinetics_id_to_classname
        eating_actions = ["eating", "tasting food"]
        drinking_actions = ["drinking", "tasting beer"]
        smoking_actions = "smoking"
        self._eating_actions = [k for k, v in kinetics_id_to_classname.items() if
                                any(action in v for action in eating_actions)]
        self._drinking_actions = [k for k, v in kinetics_id_to_classname.items() if
                                  any(action in v for action in drinking_actions)]
        self._smoking_actions = [k for k, v in kinetics_id_to_classname.items() if smoking_actions in v]

    def _load_detectron2(self):
        """
        load the detectron2 person detector model
        return: detectron2 model
        """
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._person_bbox_threshold  # set threshold for this model (default 0.55)
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = self._device
        self._person_predictor = DefaultPredictor(cfg)

    def _get_person_bboxes(self) -> np.ndarray:
        """
        generate bounding boxes for people in video using predictor
        return: predicted_boxes np.ndarray [[x_1, y_1, x_2, y_2], [x_1, y_1, x_2, y_2], ...]
        """
        # key_frame used to predict bboxes is the frame at the middle of the clip
        key_frame = self._video_data[:, self._video_data.shape[1] // 2, :, :]
        key_frame = key_frame.permute(1, 2, 0)
        predictions = self._person_predictor(key_frame.cpu().detach().numpy())['instances'].to('cpu')
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
        predicted_boxes = boxes[np.logical_and(classes == 0, scores > self._person_bbox_threshold)].tensor.cpu()
        predicted_boxes = predicted_boxes.round().numpy().astype(int)
        return predicted_boxes

    def _create_slowfast_transform(self):
        class PackPathway(torch.nn.Module):
            """
            Transform for converting video frames as a list of tensors.
            """

            def __init__(self):
                super().__init__()

            def forward(self, frames: torch.Tensor):
                fast_pathway = frames
                # Perform temporal sampling from the fast pathway.
                slow_pathway = torch.index_select(
                    frames,
                    1,
                    torch.linspace(
                        0, frames.shape[1] - 1, frames.shape[1] // 4  # 4 = slowfast alpha
                    ).long(),
                )
                frame_list = [slow_pathway, fast_pathway]
                return frame_list

        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32
        self._transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
                PackPathway()
            ]
        )

    def _create_mvit_transform(self):
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 224
        num_frames = 32
        self._transform = Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
            ]
        )

    def _load_video(self, video_path: str):
        """
        load video from video_path into pytorch tensor
        :param video_path: path of video to load
        :return: torch.Tensor [3, num_frames, frame_height, frame_width]
        """
        video_data = skvideo.io.vread(video_path)  # load video to np.ndarray
        video_data = np.einsum('klij->jkli', video_data)  # reorder ndarray dimensions to match pytorch
        return torch.from_numpy(video_data)  # convert ndarray to pytorch tensor

    def _crop_person(self, person_bbox: np.ndarray, output_video=False,
                     video_name: str = "person.mp4"):
        x1, y1, x2, y2 = person_bbox[0], person_bbox[1], person_bbox[2], person_bbox[3]
        person_data = self._video_data[:, :, y1:y2, x1:x2]
        if output_video:
            person_data_np = person_data.numpy()
            # save cropped video
            skvideo.io.vwrite(video_name, np.einsum('klij->lijk', person_data_np))
        return person_data

    def _get_top_k(self, k=5):
        top_scores, top_classes = torch.topk(self._preds, k=k)
        top_scores = top_scores.tolist()[0]
        top_classes = list(map(self._kinetics_id_to_classname.get, top_classes[0].tolist()))
        return dict(zip(top_classes, top_scores))

    def _process_preds(self):
        predictions = dict()
        predictions['eating'] = self._preds[0][self._eating_actions].sum().item()
        predictions['drinking'] = self._preds[0][self._drinking_actions].sum().item()
        predictions['smoking'] = self._preds[0][self._smoking_actions].sum().item()
        return {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}

    def _draw_bboxes(self, video_name: str, predicted_boxes: np.ndarray, labels):
        predicted_boxes = torch.from_numpy(predicted_boxes)
        for i in range(self._video_data.shape[1]):
            self._video_data[:, i, :, :] = draw_bounding_boxes(self._video_data[:, i, :, :], predicted_boxes,
                                                               labels=labels,
                                                               width=1, font_size=10, fill=True, colors="black")
        video_data = self._video_data.numpy()
        # save cropped video
        skvideo.io.vwrite(video_name, np.einsum('klij->lijk', video_data))

    def inference(self, video_path: str, visualize=False):
        self._video_data = self._load_video(video_path)
        predicted_boxes = self._get_person_bboxes()
        self._clip_no += 1
        if len(predicted_boxes) == 0:
            print("Skipping clip no persons detected at clip: ", video_path)
            return None

        actions = []
        for person_id, person_bbox in enumerate(predicted_boxes):  # for each person detected
            person_data = self._crop_person(person_bbox, output_video=False,
                                            video_name=f"output/video_{self._clip_no}_person{person_id}.mp4")
            if "slowfast" in self._model_name:
                person_data = self._transform(person_data)
                person_data = [i.to(self._device)[None, ...] for i in person_data]
                preds = self._model(person_data)
            else:
                person_data = self._transform(person_data)
                person_data = person_data.to(self._device)
                preds = self._model(person_data[None, ...])
            self._preds = self._post_act(preds)
            actions.append(self._process_preds())

        if visualize:
            labels = []
            for p in actions:
                labels.append(f"{list(p.keys())[0]}\n{round(100 * list(p.values())[0])}")
            self._draw_bboxes(video_path, predicted_boxes, labels)
        return actions

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
    """
    Implement action detection based on an action recognition model
    """

    def __init__(self, model_name: str = 'slowfast', person_bbox_threshold: float = 0.2, device: str = 'cpu'):
        """

        @param model_name: specify model to use either 'slowfast' or 'mvit'
        @param person_bbox_threshold: minimum confidence threshold for person bounding boxes
        @param device: device to use either 'cpu' or 'cuda'
        """
        # load pretrained model and configure corresponding transforms in self._transform
        if "slowfast" in model_name:
            self._model = slowfast_r50(pretrained=True)
            self._create_slowfast_transform()
        elif "mvit" in model_name:
            self._model = mvit_base_32x3(pretrained=True)
            self._create_mvit_transform()
        else:
            raise Exception(f'Invalid model name {model_name}')
        # set model to evaluation mode and move it to desired device
        self._model = self._model.to(device).eval()
        # load label map from json file
        self._load_label_map()
        self._device = device
        self._person_bbox_threshold = person_bbox_threshold
        self._model_name = model_name
        # load detectron2 person detector model and store it in self._person_predictor
        self._load_detectron2()
        # store the post_act function that will be used for inference
        self._post_act = torch.nn.Softmax(dim=1)
        self._video_data = None
        self._preds = None
        self._clip_no = -1

    def _load_label_map(self, path: str = "models/kinetics_classnames.json"):
        """
        load kinetics-400 label map
        @param path: path to json file containing the kinetics-400 labels
        @return: None
        """
        with open(path, "r") as f:  # load json file
            kinetics_classnames = json.load(f)

        # Create an id to label name mapping
        kinetics_id_to_classname = {}
        for k, v in kinetics_classnames.items():
            kinetics_id_to_classname[v] = str(k).replace('"', "")
        self._kinetics_id_to_classname = kinetics_id_to_classname
        eating_actions = ["eating", "tasting food"]  # define eating actions
        drinking_actions = ["drinking", "tasting beer"]  # define drinking actions
        smoking_actions = "smoking"  # define smoking actions
        # Store a list of IDs for all eating classes
        self._eating_actions = [k for k, v in kinetics_id_to_classname.items() if
                                any(action in v for action in eating_actions)]
        # Store a list of IDs for all drinking classes
        self._drinking_actions = [k for k, v in kinetics_id_to_classname.items() if
                                  any(action in v for action in drinking_actions)]
        # Store a list of IDs for all smoking classes
        self._smoking_actions = [k for k, v in kinetics_id_to_classname.items() if smoking_actions in v]

    def _load_detectron2(self):
        """
        load the detectron2 person detector model
        return: None
        """
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._person_bbox_threshold  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = self._device
        self._person_predictor = DefaultPredictor(cfg)

    def _get_person_bboxes(self) -> np.ndarray:
        """
        generate bounding boxes for people in self._video_data using self._person_predictor
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
        """
        create the slowfast video transform and store it in self._transform
        @return: None
        """

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
        """
        create the mvit video transform and store it in self._transform
        @return: None
        """
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
        """
        Crop a person given by person_bbox from self._video_data
        @param person_bbox: 1x4 numpy ndarray for person bbox in the form [x1, y1, x2, y2]
        @param output_video: Whether to write the cropped video for visualization purposes
        @param video_name: Name of output video, unused if output_video is False
        @return: cropped_video torch.Tensor [3, num_frames, height, width]
        """
        x1, y1, x2, y2 = person_bbox[0], person_bbox[1], person_bbox[2], person_bbox[3]
        person_data = self._video_data[:, :, y1:y2, x1:x2]  # slice person from video tensor
        if output_video:
            person_data_np = person_data.numpy()
            # save cropped video
            skvideo.io.vwrite(video_name, np.einsum('klij->lijk', person_data_np))
        return person_data

    def _get_top_k(self, k=5):
        """
        Get top k classes from self._preds
        @param k: number of top k classes to return
        @return: dict of top k classes in descending order of confidence {"action1": conf1, "action2": conf2,...}
        """
        top_scores, top_classes = torch.topk(self._preds, k=k)
        top_scores = top_scores.tolist()[0]
        top_classes = list(map(self._kinetics_id_to_classname.get, top_classes[0].tolist()))
        return dict(zip(top_classes, top_scores))

    def _process_preds(self):
        """
        process self._preds to combine classes of interest
        @return: sorted dict of classes {"eating": conf1, "drinking": conf2, "smoking": conf3}
        """
        predictions = dict()
        # combine all eating actions
        predictions['eating'] = self._preds[0][self._eating_actions].sum().item()
        # combine all drinking actions
        predictions['drinking'] = self._preds[0][self._drinking_actions].sum().item()
        # combine all smoking actions
        predictions['smoking'] = self._preds[0][self._smoking_actions].sum().item()
        # sort dict by confidence before returning it
        return {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}

    def _draw_bboxes(self, video_name: str, predicted_boxes: np.ndarray, labels):
        """
        Draw bounding boxes around detected persons in video then write this video to disk
        @param video_name: name of video to save
        @param predicted_boxes: predicted_boxes np.ndarray [[x_1, y_1, x_2, y_2], [x_1, y_1, x_2, y_2], ...]
        @param labels: list of strings containing label for each bounding box
        @return: None
        """
        predicted_boxes = torch.from_numpy(predicted_boxes)  # convert bboxes to tensor
        for i in range(self._video_data.shape[1]):  # loop over all frames and draw bbox in each one
            self._video_data[:, i, :, :] = draw_bounding_boxes(self._video_data[:, i, :, :], predicted_boxes,
                                                               labels=labels,
                                                               width=1, font_size=10, fill=True, colors="black")
        video_data = self._video_data.numpy()  # convert video to numpy
        # save video
        skvideo.io.vwrite(video_name, np.einsum('klij->lijk', video_data))

    def inference(self, video_path: str, visualize=False):
        """
        Perform action detection inference on video
        @param video_path: path of query video
        @param visualize: if true an output video containing action detection visualization is written to disk
        @return: list of dictionaries where each dictionary contains action confidence pairs for each person detected
        """
        self._video_data = self._load_video(video_path)  # load video to tensor
        predicted_boxes = self._get_person_bboxes()  # get bboxes of persons in video
        self._clip_no += 1
        if len(predicted_boxes) == 0:  # if no persons detected in video skip it
            print("Skipping clip no persons detected at clip: ", video_path)
            return None

        actions = []
        for person_id, person_bbox in enumerate(predicted_boxes):  # for each person detected
            # crop person from video
            person_data = self._crop_person(person_bbox, output_video=False,
                                            video_name=f"output/video_{self._clip_no}_person{person_id}.mp4")
            # transform video data before feeding into the model
            person_data = self._transform(person_data)
            if "slowfast" in self._model_name:
                # move video to device before feeding into the model
                person_data = [i.to(self._device)[None, ...] for i in person_data]
                preds = self._model(person_data)  # perform inference
            else:
                # move video to device before feeding into the model
                person_data = person_data.to(self._device)
                preds = self._model(person_data[None, ...])  # perform inference
            self._preds = self._post_act(preds)  # apply softmax to predictions
            actions.append(self._process_preds())  # process predictions and append new results to list
            # actions.append(self._get_top_k()) # get top k predictions and append new results to list

        if visualize:  # if visualization is enabled
            labels = []  # create labels from top 1 and confidence of each person
            for p in actions:
                labels.append(f"{list(p.keys())[0]}\n{round(100 * list(p.values())[0])}")
            # output video with bboxes and labels
            self._draw_bboxes(video_path, predicted_boxes, labels)
        return actions

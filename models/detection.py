import numpy as np

import cv2
import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from models.visualization import VideoVisualizer
import pytorchvideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection  # Another option is slowfast_r50_detection




class DetectionModel:
    def __init__(self, person_bbox_threshold: float = 0.2, device: str = 'cpu'):
        self._model = slowfast_r50_detection(True)  # Another option is slowfast_r50_detection
        self._model = self._model.eval().to(device)
        self._label_map, self._allowed_class_ids = AvaLabeledVideoFramePaths.read_label_map(
            'models/ava_action_list_mod.pbtxt')
        self._device = device
        self._person_bbox_threshold = person_bbox_threshold
        self._load_detectron2()
        self._video_data = None
        self._preds = None
        self._clip_no = -1

    def _load_detectron2(self):
        """
        load the detectron2 person detector model
        return: None
        """
        cfg = get_cfg()
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._person_bbox_threshold  # set threshold for this model
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.MODEL.DEVICE = self._device
        self._person_predictor = DefaultPredictor(cfg)

    def _get_person_bboxes(self, inp_img):
        """
        generate bounding boxes for people in image using predictor
        @param inp_img: image
        @param predictor: predictor model to use
        @param threshold: minimum confidence threshold used for predictions
        return: bounding boxes
        """
        predictions = self._person_predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
        predicted_boxes = boxes[np.logical_and(classes == 0, scores > self._person_bbox_threshold)].tensor.cpu()
        return predicted_boxes

    def _ava_inference_transform(self,
                                 clip,
                                 boxes,
                                 num_frames=32,  # 4 def, if using slowfast_r50_detection, change this to 32
                                 crop_size=256,
                                 data_mean=[0.45, 0.45, 0.45],
                                 data_std=[0.225, 0.225, 0.225],
                                 slow_fast_alpha=4,  # none def, if using slowfast_r50_detection, change this to 4
                                 ):
        boxes = np.array(boxes)
        ori_boxes = boxes.copy()

        # Image [0, 255] -> [0, 1].
        clip = uniform_temporal_subsample(clip, num_frames)
        clip = clip.float()
        clip = clip / 255.0

        height, width = clip.shape[2], clip.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, width] for x and [0,height] for y
        boxes = clip_boxes_to_image(boxes, height, width)

        # Resize short side to crop_size. Non-local and STRG uses 256.
        clip, boxes = short_side_scale_with_boxes(
            clip,
            size=crop_size,
            boxes=boxes,
        )

        # Normalize images by mean and std.
        clip = normalize(
            clip,
            np.array(data_mean, dtype=np.float32),
            np.array(data_std, dtype=np.float32),
        )

        boxes = clip_boxes_to_image(
            boxes, clip.shape[2], clip.shape[3]
        )

        # Incase of slowfast, generate both pathways
        if slow_fast_alpha is not None:
            fast_pathway = clip
            # Perform temporal sampling from the fast pathway.
            slow_pathway = torch.index_select(
                clip,
                1,
                torch.linspace(
                    0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
                ).long(),
            )
            clip = [slow_pathway, fast_pathway]

        return clip, torch.from_numpy(boxes), ori_boxes

    def inference(self, video_path: str, visualize=False):
        encoded_vid = pytorchvideo.data.encoded_video.EncodedVideo.from_path(video_path, decode_audio=False)
        print("Generating predictions for clip: {}".format(video_path))
        # Generate clip around the designated time stamps
        inp_imgs = encoded_vid.get_clip(0, encoded_vid.duration)
        inp_imgs = inp_imgs['video']

        # Generate people bbox predictions using Detectron2's off the self pre-trained predictor
        # We use the the middle image in each clip to generate the bounding boxes.
        inp_img = inp_imgs[:, inp_imgs.shape[1] // 2, :, :]
        inp_img = inp_img.permute(1, 2, 0)

        # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
        predicted_boxes = self._get_person_bboxes(inp_img)
        if len(predicted_boxes) == 0:
            print("Skipping clip no persons detected at clip: ", video_path)
            return None

        # Preprocess clip and bounding boxes for video action recognition.
        inputs, inp_boxes, _ = self._ava_inference_transform(inp_imgs, predicted_boxes.numpy())
        # Prepend data sample id for each bounding box.
        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)

        # Generate actions predictions for the bounding boxes in the clip.
        # The model here takes in the pre-processed video clip and the detected bounding boxes.
        if isinstance(inputs, list):
            inputs = [inp.unsqueeze(0).to(self._device) for inp in inputs]
        else:
            inputs = inputs.unsqueeze(0).to(self._device)
        preds = self._model(inputs, inp_boxes.to(self._device))
        preds = preds[:, [4, 26, 28, 53]]
        # get top k predictions and corresponding scores for each bounding box
        top_scores, top_classes = torch.topk(preds, k=4)
        # convert predictions and scores to list
        top_scores, top_classes = top_scores.tolist(), top_classes.tolist()
        persons = []
        for person_top_classes, person_top_scores in zip(top_classes,
                                                         top_scores):  # loop over each person's actions scores pairs
            person_top_classes = [self._label_map.get(top_class + 1, "N/A" + str(top_class)) for top_class in
                                  person_top_classes]
            current_person = dict(zip(person_top_classes, person_top_scores))
            persons.append(current_person)

        if visualize:  # output video visualization
            preds = preds.to('cpu')
            preds = torch.cat([torch.zeros(preds.shape[0], 1), preds], dim=1)
            inp_imgs = inp_imgs.permute(1, 2, 3, 0)
            inp_imgs = inp_imgs / 255.0
            video_visualizer = VideoVisualizer(4, self._label_map, top_k=4, mode="top-k", thres=0)
            gif_imgs = video_visualizer.draw_clip_range(inp_imgs, preds * 100, predicted_boxes)
            height, width = gif_imgs[0].shape[0], gif_imgs[0].shape[1]

            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 7, (width, height))

            for image in gif_imgs:
                img = (255 * image).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)
            video.release()
        return persons

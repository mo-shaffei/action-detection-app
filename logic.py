from datetime import datetime, timedelta
import helpers
from models.recognizer import RecognizerModel
from vidgear.gears import NetGear
import numpy as np
import threading

port_mapping = {
    5554: ('Helmy', 'Computer Lab S008', 'CAM01'),
    5555: ('Nano', 'Computer Lab S012A', 'CAM01')
}

action_confs_map = {
    'drinking': 0.10,
    'eating': 0.25,
    'smoking': 0.8
}


def predict_stream(port, segment_len, stride, model, visualize, fps=25):
    client = NetGear(port=str(port), receive_mode=True)
    metadata = port_mapping[port]
    building, area, camera_id = metadata
    frame = client.recv()
    video = []
    accumulated = 1
    required = fps * segment_len
    begin = datetime.now()
    while True:
        if frame is None:
            break
        video.append(frame)
        accumulated += 1
        print('Received frame, accumulated: ', accumulated)
        if accumulated == required:
            print('Performing Inference')
            persons = model.inference(np.asarray(video), visualize=visualize)
            accumulated = 0
            video = []
            if not persons:  # if no persons detected skip this segment
                print('Nothing detected for this clip')
                continue

            for person in persons:  # for each person detected
                action = list(person.keys())[0]  # get top1 action
                confidence = list(person.values())[0]  # get confidence of top1 action
                print(f'Detected {action}, {confidence}, {action_confs_map[action]}')
                if confidence >= action_confs_map[action]:  # only store action if confidence >= threshold
                    end = begin + timedelta(seconds=segment_len)  # end time of segment = beg time + segment length
                    helpers.output(begin, end, action, confidence, building, area, camera_id)
            begin = datetime.now()
        frame = client.recv()


def predict_video(video_name, segment_len, stride, model, visualize):
    path = 'video/'  # store path of videos
    # segment the input video into multiple segments as required by segment_len and stride, return the resulting
    # number of segments
    segments = helpers.video2segments(path, video_name, segment_len=segment_len, stride=stride)
    # No stream = Debug mode so set metadata arbitrarily
    building = 'Helmy'
    area = 'Computer Lab S008'
    camera_id = 'CAM01'
    begin = datetime.now()
    for i in range(segments):  # process segment by segment
        print(f"progress: {int(i * 100 / segments)}%")  # print progress to terminal
        # perform inference on current segment
        persons = model.inference(path + f"video_{i}.mp4", visualize=visualize)
        if not persons:  # if no persons detected skip this segment
            print('Nothing detected for this clip')
            continue

        for person in persons:  # for each person detected
            action = list(person.keys())[0]  # get top1 action
            confidence = list(person.values())[0]  # get confidence of top1 action
            print(f'Detected {action}, {confidence}, {action_confs_map[action]}')
            if confidence >= action_confs_map[action]:  # only store action if confidence >= threshold
                beg = begin + timedelta(seconds=i * stride)
                end = beg + timedelta(seconds=segment_len)  # end time of segment = beg time + segment length
                helpers.output(beg, end, action, confidence, building, area, camera_id)

    print("Inference thread finished!")


def inference_thread(app, ports, use_stream=False):
    """
    Wrapper around connect button function to make it run in a separate thread
    @param app: flask app (used to extract configurations for model and video)
    @param use_stream: whether to use stream or read from video files
    @param port: specify port of streaming server if using stream
    @return: None
    """

    print(f"Inference thread {ports[0]} started...")
    # minimum confidence threshold for top1 action to store it
    confidence_thresh = app.config["model"]["recognition_threshold"]
    # stride length (in seconds) of temporal window which segments the input video
    stride = app.config["video"]["stride"]
    # segment length (in seconds) of each mini video segment
    segment_len = app.config["video"]["clip_length"]
    bbox_threshold = app.config["model"]["bbox_threshold"]  # store person bbox minimum confidence threshold
    visualize = app.config["model"]["visualize"]  # store whether to visualize results or not
    device = app.config["model"]["device"]  # store device used for inference
    model_name = app.config["model"]["model_name"]

    if use_stream:
        for port in ports:
            model = RecognizerModel(model_name=model_name, person_bbox_threshold=bbox_threshold, device=device)
            args = [port, segment_len, stride, model, visualize]
            t = threading.Thread(target=predict_stream, args=args)
            t.setDaemon(True)
            t.start()
    else:
        video_name = app.config["video"]["video_name"]  # store name of video to process
        model = RecognizerModel(model_name=model_name, person_bbox_threshold=bbox_threshold, device=device)
        predict_video(video_name, segment_len, stride, model, visualize)

import pymongo
import helpers
from models.recognizer import RecognizerModel
from models.detection import DetectionModel
import time


def connect_thread(app):
    """
    wrapper around connect button function to make it run in a separate thread
    """

    print("thread started")
    # minimum confidence threshold in top1 action to store it
    print(app.config)
    confidence_thresh = app.config["model"]["recognition_threshold"]
    # stride length (in seconds) of temporal window which segments the input video
    stride = app.config["video"]["stride"]
    # segment length (in seconds) of each mini video segment
    segment_len = app.config["video"]["clip_length"]
    path = 'video/'  # store path of videos
    video_name = app.config["video"]["video_name"]
    # segment the input video into multiple segments as required by segment_len and stride, return the resulting
    # number of segments
    segments = helpers.video2segments(path, video_name, segment_len=segment_len, stride=stride)
    # segments = 10

    previous_action = " "
    previous_beg = 0
    previous_confidence = 0
    clips = []
    bbox_threshold = app.config["model"]["bbox_threshold"]
    visualize = app.config["model"]["visualize"]
    device = app.config["model"]["device"]
    if app.config["model"]["model_type"] == "recognition":
        model_name = app.config["model"]["model_name"]
        model = RecognizerModel(model_name=model_name, person_bbox_threshold=bbox_threshold, device=device)
    else:
        model = DetectionModel(person_bbox_threshold=bbox_threshold, device=device)
    start_time = time.time()
    for i in range(segments):  # process segment by segment
        print(f"progress: {int(i * 100 / segments)}%")  # print progress to terminal
        # persons = helpers.inference(path + f"video_{i}.mp4", segment_len)  # perform inference on current segment
        persons = model.inference(path + f"video_{i}.mp4", visualize=visualize)
        if not persons:
            continue

        for person in persons:
            action = list(person.keys())[0]  # get top1 action
            confidence = list(person.values())[0]  # get confidence of top1 action
            if confidence >= confidence_thresh:  # only store action if confidence >= threshold

                # if action == previous_action:
                #     last_row = app.results_data.find_one(
                #         {}, sort=[('_id', pymongo.DESCENDING)])
                #     app.results_data.delete_one(last_row)
                #     beg = previous_beg
                #     end = i * stride + segment_len
                #     confidence = (confidence + previous_confidence) / 2
                #     clips.append(i)
                #     helpers.output(beg, end, action, confidence, clips)
                #
                # else:
                beg = i * stride  # beg time of segment = current segment index * stride length
                end = beg + segment_len  # end time of segment = beg time + segment length
                # clips = [i]
                helpers.output(beg, end, action, confidence, i)

                # previous_action = action
                # previous_beg = beg
                # previous_confidence = confidence
    stop_time = time.time()
    print("Done!")
    print(f"Time elapsed: {stop_time - start_time}")

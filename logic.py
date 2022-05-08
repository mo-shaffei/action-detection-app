import helpers
#from models.recognizer import RecognizerModel
#from models.detection import DetectionModel
import time


def connect_thread(app):
    """
    Wrapper around connect button function to make it run in a separate thread
    @param app: flask app (used to extract configurations for model and video)
    @return: None
    """

    print("Inference thread started...")
    # minimum confidence threshold for top1 action to store it
    confidence_thresh = app.config["model"]["recognition_threshold"]
    # stride length (in seconds) of temporal window which segments the input video
    stride = app.config["video"]["stride"]
    # segment length (in seconds) of each mini video segment
    segment_len = app.config["video"]["clip_length"]
    path = 'video/'  # store path of videos

    video_name = app.config["video"]["video_name"]  # store name of video to process
    # segment the input video into multiple segments as required by segment_len and stride, return the resulting
    # number of segments
    segments = helpers.video2segments(path, video_name, segment_len=segment_len, stride=stride)

    bbox_threshold = app.config["model"]["bbox_threshold"]  # store person bbox minimum confidence threshold
    visualize = app.config["model"]["visualize"]  # store whether to visualize results or not
    device = app.config["model"]["device"]  # store device used for inference
    if app.config["model"]["model_type"] == "recognition":  # load appropriate model
        model_name = app.config["model"]["model_name"]
        model = RecognizerModel(model_name=model_name, person_bbox_threshold=bbox_threshold, device=device)
    else:
        model = DetectionModel(person_bbox_threshold=bbox_threshold, device=device)

    # For now we will define a constant camera id and location
    camera_id = "NB-1001"
    location = "Studio"
    start_time = time.time()
    for i in range(segments):  # process segment by segment
        print(f"progress: {int(i * 100 / segments)}%")  # print progress to terminal
        # perform inference on current segment
        persons = model.inference(path + f"video_{i}.mp4", visualize=visualize)
        if not persons:  # if no persons detected skip this segment
            continue

        for person in persons:  # for each person detected
            action = list(person.keys())[0]  # get top1 action
            confidence = list(person.values())[0]  # get confidence of top1 action
            if confidence >= confidence_thresh:  # only store action if confidence >= threshold
                beg = i * stride  # beg time of segment = current segment index * stride length
                end = beg + segment_len  # end time of segment = beg time + segment length
                helpers.output(camera_id, beg, end, action, confidence, i, location)

    stop_time = time.time()
    print("Inference thread finished!")
    print(f"Time elapsed: {stop_time - start_time}")

import pymongo
import helpers
import app
from models.recognizer import RecognizerModel
from models.detection import DetectionModel


def connect_thread():
    """
    wrapper around connect button function to make it run in a separate thread
    """

    print("thread started")
    confidence_thresh = 0  # minimum confidence threshold in top1 action to store it
    stride = 2  # stride length (in seconds) of temporal window which segments the input video
    segment_len = 3  # segment length (in seconds) of each mini video segment
    path = 'video/'  # store path of videos
    # segment the input video into multiple segments as required by segment_len and stride, return the resulting
    # number of segments
    segments = helpers.video2segments(path, "video.mp4", segment_len=segment_len, stride=stride)
    # segments = 10

    previous_action = " "
    previous_beg = 0
    previous_confidence = 0
    clips = []
    model = RecognizerModel(model_name='mvit', person_bbox_threshold=0.5, device='cpu')
    # model = DetectionModel(person_bbox_threshold=0.35, device='cpu')
    for i in range(segments):  # process segment by segment
        print(f"progress: {int(i * 100 / segments)}%")  # print progress to terminal
        # persons = helpers.inference(path + f"video_{i}.mp4", segment_len)  # perform inference on current segment
        persons = model.inference(path + f"video_{i}.mp4", visualize=True)
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
    print("Done!")

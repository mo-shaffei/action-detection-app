import pymongo
import helpers
import app


def connect_thread():
    """
    wrapper around connect button function to make it run in a separate thread
    """

    print("thread started")
    confidence_thresh = 0.2  # minimum confidence threshold in top1 action to store it
    stride = 5  # stride length (in seconds) of temporal window which segments the input video
    segment_len = 10  # segment length (in seconds) of each mini video segment
    path = 'video/'  # store path of videos
    # segment the input video into multiple segments as required by segment_len and stride, return the resulting
    # number of segments
    segments = helpers.video2segments(path, "video.mp4", segment_len=segment_len, stride=stride)
    # segments = 10

    previous_action = " "
    previous_beg = 0
    previous_confidence = 0
    clips = []

    for i in range(segments):  # process segment by segment
        print(f"progress: {int(i * 100 / segments)}%")  # print progress to terminal
        response = helpers.inference(path + f"video_{i}.mp4", "slowfast_rec")  # perform inference on current segment
        action = list(response.keys())[0]  # get top1 action
        confidence = list(response.values())[0]  # get confidence of top1 action
        if confidence >= confidence_thresh:  # only store action if confidence >= threshold

            if action == previous_action:
                last_row = app.results_data.find_one(
                    {}, sort=[('_id', pymongo.DESCENDING)])
                app.results_data.delete_one(last_row)
                beg = previous_beg
                end = i * stride + segment_len
                confidence = (confidence + previous_confidence) / 2
                clips.append(i)
                helpers.output(beg, end, action, confidence, clips)

            else:
                beg = i * stride  # beg time of segment = current segment index * stride length
                end = beg + segment_len  # end time of segment = beg time + segment length
                clips = [i]
                helpers.output(beg, end, action, confidence, i)

            previous_action = action
            previous_beg = beg
            previous_confidence = confidence

    print("done!")

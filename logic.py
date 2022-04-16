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
    # with open(path + "results.csv", 'w') as file:  # create results file and add column headers to it
    #    file.write("start,end,action,confidence,clip\n")

    for i in range(segments):  # process segment by segment
        print(f"progress: {int(i * 100 / segments)}%")  # print progress to terminal
        response = helpers.inference(path + f"video_{i}.mp4", "slowfast_rec")  # perform inference on current segment
        action = list(response.keys())[0]  # get top1 action
        confidence = list(response.values())[0]  # get confidence of top1 action
        if confidence >= confidence_thresh:  # only store action if confidence >= threshold
            beg = i * stride  # beg time of segment = current segment index * stride length
            end = beg + segment_len  # end time of segment = beg time + segment length
            # with open(path + "/results.csv", 'a') as file:  # append data to the results file
            # helpers.output(1, 2, "eatinggg", 23, i)
            helpers.output(beg, end, action, confidence, i)

    print("done!")
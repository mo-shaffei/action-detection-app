import requests
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import subprocess
import app
import model

detectron = model.load_detectron2()
slowfast = model.load_slowfast()


def get_length(path: str) -> int:
    """
    return the duration (in seconds) of the video given by the path
    """
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return int(float(result.stdout))


def video2segments(path: str, filename: str, segment_len: int = 10, stride: int = 5) -> int:
    """
    split video given by path into segments
    """
    duration = get_length(path + filename)
    c = 0
    for i in range(0, duration, stride):
        ffmpeg_extract_subclip(path + filename, i, i + segment_len, targetname=path + f"video_{c}.mp4")
        c += 1
    return c


def inference(path: str, video_duration: int) -> dict:
    persons = model.get_actions(path, slowfast, detectron, top_k=1, visualize=False)
    return persons


def output(time_beg: int, time_end: int, action: str, confidence: float, reference: int) -> None:
    confidence = round(confidence * 100)
    app.results_data.insert_one({"start": time_beg, "end": time_end,
                                 "action": action, "confidence": confidence,
                                 "clip": reference})  # inserting results into database

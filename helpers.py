from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip
import subprocess
import app


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


def video2segments2(path: str, filename: str, segment_len: int = 10, stride: int = 5) -> int:
    """
    (OLD FUNCTION, not used as it produced some bugs)
    split video given by path into segments
    """
    duration = get_length(path + filename)
    c = 0
    for i in range(0, duration, stride):
        ffmpeg_extract_subclip(path + filename, i, i + segment_len, targetname=path + f"video_{c}.mp4")
        c += 1
    return c


def video2segments(path: str, filename: str, segment_len: int = 10, stride: int = 5) -> int:
    """
    split video given by path into segments
    """
    with VideoFileClip(path + filename) as video:
        c = 0
        for i in range(0, int(video.duration), stride):
            new = video.subclip(i, i + segment_len)
            new.write_videofile(path + f"video_{c}.mp4")
            c += 1
    return c


def output(camera_id: str, time_beg: int, time_end: int, action: str,
           confidence: float, reference: int, location: str) -> None:
    confidence = round(confidence * 100)
    app.results_data.insert_one({"camera_id": camera_id, "start": time_beg, "end": time_end,
                                 "action": action, "confidence": confidence,
                                 "clip": reference, "location": location})  # inserting results into database

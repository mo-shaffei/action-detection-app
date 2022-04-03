import requests
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import subprocess
from django.conf import settings


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


def video2segments(path: str, segment_len: int = 10, stride: int = 5) -> int:
    """
    split video given by path into segments
    """
    duration = get_length(path)
    c = 0
    path2 = str(settings.BASE_DIR) + '/pages'
    for i in range(0, duration, stride):
        ffmpeg_extract_subclip(path, i, i + segment_len, targetname=path2 + f"/video_{c}.mp4")
        c += 1
    return c


def inference(path: str, model_name: str) -> dict:
    url = 'http://127.0.0.1:8080/predictions/' + model_name
    response = requests.put(url, data=open(path, 'rb').read())
    return response.json()


def output(file, time_beg: int, time_end: int, action: str, confidence: float, reference: int) -> None:
    file.write(f'{time_beg},{time_end},{action},{confidence}, {reference}\n')

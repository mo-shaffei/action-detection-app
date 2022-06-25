# import libraries
import time
import threading
import sys
import cv2
from vidgear.gears import VideoGear
from vidgear.gears import NetGear


FPS = 25


def streaming_thread(port, video):
    stream = VideoGear(source=video).start()  # Open any video stream
    options = {'request_timeout': 100}
    server = NetGear(port=port, **options)  # Define netgear server with default settings
    frame_count = 1
    while True:
        try:
            frame = stream.read()
            # check if frame is None
            if frame is None:
                # if True break the infinite loop
                break

            server.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print(f"Sent frame {frame_count} on {port}")
            frame_count += 1

            time.sleep(1 / FPS)

        except:
            # break the infinite loop
            break

    # safely close video stream
    stream.stop()
    server.close()


def main(argv):
    n = int(argv[1])
    ports = argv[2:2 + n]
    videos = argv[2 + n:]
    print(n, ports, videos)
    threads = []
    for i in range(n):
        t = threading.Thread(target=streaming_thread, args=[ports[i], videos[i]])
        t.setDaemon(True)
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv)

__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import json
import subprocess
import time

import cv2

from inference_video import inf_run


def probe_stream(video_path, cam_name):
    # client 10.61.172.166 (and or 10.61.41.4) can set RST in the TCP header if server 10.61.41.4 does not send
    # payload after 10.61.172.166 ACK. Means the streaming server has an issue

    cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
    result = cap.isOpened()

    if result:
        inf_run(cam_name, video_path)

    elif not result:
        # assumes if there is no stream from a single camera, all cameras do not work.
        # all streams are from the same server, means issue with server.
        from main import main
        print(f"Camera {cam_name} not available, restarting main in 61 seconds.....")
        time.sleep(61)
        print('Restarting main........')
        main()



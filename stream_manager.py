__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import cv2
from inference_video import inf_run


# client 10.61.172.166 (and or 10.61.41.4) can set RST in the TCP header if server 10.61.41.4 does not send
# payload after 10.61.172.166 ACK. Means the streaming server has an issue or the network itself
# when the stream starts again, opencv does not appear to detect it, so restarting the application after 61 secs
def probe_stream(video_path, cam_name):
    cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
    result = cap.isOpened()
    cap.release()  # Release the capture object

    # assumes if there is no stream from a single camera, all cameras do not work (appears to be the case)
    # all streams are from the same server, means issue with the server on network itself.
    if result:
        inf_run(cam_name, video_path)
        return True
    else:
        return False

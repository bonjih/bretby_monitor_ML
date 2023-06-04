__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import subprocess
import cv2

from inference_video import inf_run


# client 10.61.172.166 (and or 10.61.41.4) can set RST in the TCP header if server 10.61.41.4 does not send
# payload after 10.61.172.166 ACK. Means the streaming server has an issue or the network itself
# when the stream starts again, opencv does not appear to detect it, so restarting the application after 61 secs
# by calling the script nssm service


def restart_nssm_service(service_name):
    command = fr'C:\nssm224\win64\nssm.exe restart {service_name}'
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while restarting the service: {e}")


def probe_stream(video_path, cam_name):
    cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
    result = cap.isOpened()
    cap.release()

    if result:
        inf_run(cam_name, video_path)
        return True
    else:
        return False

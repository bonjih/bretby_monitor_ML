import time
import cv2
from datetime import datetime


def vid_save(fps, frame_width, frame_height, cam_name):
    now = datetime.now()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoOut = cv2.VideoWriter(f'C:\\bretby_monitor_ML\\saved_videos\\{cam_name}.mp4', fourcc, fps,
                               (frame_width, frame_height))
    return videoOut


def save_image(cam_name, image):
    """
    saves an image if condition (result) is True
    :return:
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(f'C:\\bretby_monitor_ML\\saved_images\\{cam_name}-{timestr}.jpg', image)

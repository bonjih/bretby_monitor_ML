import cv2
from datetime import datetime


def vid_save(fps, frame_width, frame_height, cam_name):
    now = datetime.now()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoOut = cv2.VideoWriter(f'saved_videos/{cam_name}.mp4', fourcc, fps, (frame_width, frame_height))
    return videoOut
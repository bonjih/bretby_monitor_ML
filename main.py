__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import os
import sys
import time
import pandas as pd

import global_conf_variables
from inference_video import inf_run
from stream_manager import probe_stream

values = global_conf_variables.get_values()

cams = values[0]


def restart():
    os.execl(sys.executable, sys.executable, *sys.argv)


def run(cam_name, camID):
    #cap = cv2.VideoCapture('rtsp://10.61.172.92:11011/grv1/shield/033/hd')
    inf_run(cam_name, camID)
    # if cap.isOpened():
    #
    #
    # else:
    #     print("[NO STREAM FROM]" + camID)


def main():
    # to track if a variable has not been cleared
    # if not cleared after 61 secs, restart main

    while True:
        # try:
        df = pd.read_csv(cams)

        print('Looping through camera list......\n')

        for index, row in df.iterrows():

            print(row['cam_name'], '->', row['address'])
            result = probe_stream(row['address'], row['cam_name'])  # comment out when using MP4
            print(result)

            # result = True  # uncomment when using MP4
            if result is not None or result:
                run(row['cam_name'], row['address'])
                # time.sleep(2)
            else:
                print(f"Camera {row['cam_name']} not available, moving to next camera...")

            # except Exception as e:
            #     print(e)


if __name__ == "__main__":
    main()

__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import re
import time
import pandas as pd

bret_coords_all = []
bret_data = []


def format_df():
    if len(bret_data) != 0:
        data_all = [bret_data[i:i + 6] for i in range(0, len(bret_data), 6)]
        df = pd.DataFrame(data_all, columns=['x', 'y', 'CameraName', 't0', 't1', 'PercentageTroughFull'])
        return df


def format_data(i, cam_name, res, tup_1, tup_2, percent, trail_history):
    if not res:
        bret_coords_all.extend([trail_history[i][0][0], cam_name, time.time()])
        t0 = bret_coords_all[2]
        t1 = bret_coords_all[-1]
        bret_coords_all.append(t1 - t0)
        x, y = re.findall(r'(\d+)', str(tup_2))
        camera, time_0, time_1 = re.findall(r'[A-Za-z]+', str(tup_1))
        bret_data.extend([x, y, camera, time_0, time_1, percent])

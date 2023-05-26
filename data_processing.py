__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import os
from datetime import datetime
import pandas as pd

import global_conf_variables
from db_manager import db_manager_controller
from model.utils.transforms import is_similar, convert_img_for_db
from utils.save_file_methods import save_image

values = global_conf_variables.get_values()

# % value of debris in the trough, used as a flag to save image
pct_of_debris = float(values[4])

# an arbitrary buffer to x/y of bretby to cater for exceptions
x_more = -0.39
y_more = -0.1


def add_x(sum_x):
    """
    adds a proportion to the values in the x
    :return: values x
    """
    result_x = sum_x * (1 + x_more)
    return result_x


def add_y(sum_y):
    """
    adds a proportion to the values in the y
    :return: values y
    """
    result_y = sum_y * (1 + y_more)
    return result_y


def greater_x(x, add_x):
    """
    return True if 'normal' x pix value is less than y pix value
    meaning bretby has moved up the trough
    :param x:
    :param add_x:
    :return True/False:
    """
    return x < add_x


def greater_y(y, add_y):
    """
    return True if 'normal' y pix value is less than y pix value
    meaning bretby has moved up the trough
    :param x:
    :param add_x:
    :return True/False:
    """
    return y < add_y


def check_eqal(result_x, result_y):
    return result_x != result_y


def get_time_diff(df):
    """
    get time difference between ball movement
    :return: time in ps
    """
    df['TimeStamp'] = (df['t0'].astype(float) - df['t1'].astype(float)) / 100000000
    df.TimeStamp = pd.to_datetime(df.TimeStamp, unit='ps')
    df['diff'] = (df['TimeStamp'] - df['TimeStamp'].shift(1))
    return df


def check_saved_image(cam_name, img):
    save_dir = 'C:\\bretby_monitor_ML\\saved_images'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    prev_img_path = os.path.join(save_dir, 'prev_img.jpg')
    if not os.path.isfile(prev_img_path):
        save_image(cam_name, img)
        bts_img = convert_img_for_db(img)
        df_bts_img = pd.DataFrame([bts_img])
        df_bts_img.rename(columns={0: 'Image'}, inplace=True)
        return df_bts_img

    result = is_similar(img, prev_img_path)
    if result:
        save_image(cam_name, img)
        bts_img = convert_img_for_db(img)
        df_bts_img = pd.DataFrame([bts_img])
        df_bts_img.rename(columns={0: 'Image'}, inplace=True)
        return df_bts_img


def bret_loc_data(df_infer, cam_name, img, bb_results):
    try:
        if not df_infer.empty or not None:
            if bb_results:
                df = df_infer
                if not df.empty:
                    df['BretbyDebLoc_x'] = df.apply(lambda row: add_x(float(row['x'])), axis=1)
                    df['BretbyDebLoc_y'] = df.apply(lambda row: add_y(float(row['y'])), axis=1)
                    df['result_x'] = df.apply(lambda row: greater_x(float(row['x']), float(row['BretbyDebLoc_x'])),
                                              axis=1)
                    df['result_y'] = df.apply(lambda row: greater_y(float(row['y']), float(row['BretbyDebLoc_y'])),
                                              axis=1)

                    df['DebrisResult'] = df['PercentageTroughFull'] > pct_of_debris
                    df['BretbyResult'] = df.apply(lambda row: check_eqal(row['result_x'], row['result_y']), axis=1)

                    df = df[df['result_x'] == False]

                    df_bts_img = check_saved_image(cam_name, img)
                    df = df.join(df_bts_img)

                    df_pass = df[['CameraName', 'PercentageTroughFull', 'BretbyDebLoc_x', 'BretbyDebLoc_y',
                                  'DebrisResult', 'BretbyResult', 'Image']]

                    df_pass_true = df_pass[df_pass['DebrisResult']]

                    df_pass_true.to_csv('temp_out.csv', index=False)
                    db_manager_controller()
                    df_pass_true.loc[:] = None
    except Exception as e:
        print(e, 'data-pro - ', datetime.now())


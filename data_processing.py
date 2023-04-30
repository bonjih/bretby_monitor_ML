__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import glob
import os

import pandas as pd

import global_conf_variables
from db_manager import db_manager_controller
from model.utils.transforms import save_image, is_similar, convert_img_for_db

values = global_conf_variables.get_values()

pct_of_debris = float(values[4])

# an arbitrary buffer to x/y of bretby to cater for exceptions
x_more = -0.39
y_more = -0.1


def check_if_df_empty(df):
    return len(df.index) == 0


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
    meaning breby has moved up the trough
    :param x:
    :param add_x:
    :return True/False:
    """
    if x < add_x:
        return True
    else:
        return False


def greater_y(y, add_y):
    """
    return True if 'normal' y pix value is less than y pix value
    meaning breby has moved up the trough
    :param x:
    :param add_x:
    :return True/False:
    """
    if y < add_y:
        return True
    else:
        return False


def check_eqal(result_x, result_y):
    if result_x != result_y:
        return True
    elif result_x == result_x:
        return False


def get_time_diff(df):
    """
    get time difference between ball movement
    :return: time in ps
    """

    df['TimeStamp'] = (df['t0'].astype(float) - df['t1'].astype(float)) / 100000000
    df.TimeStamp = pd.to_datetime(df.TimeStamp, unit='ps')
    df['diff'] = (df['TimeStamp'] - df['TimeStamp'].shift(1))
    # df.drop('TimeStamp', axis=1, inplace=True)
    # print(df['diff'])
    return df


def get_change_in_xy(df):
    df = df.copy()
    df['diff_x'] = abs((df['x'].astype(int) - df['x'].astype(int).shift(1)))
    df['diff_y'] = abs((df['y'].astype(int) - df['y'].astype(int).shift(1)))

    # df = df.loc[df['diff_y'] != 0.0].copy()
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    return df


# first saves an image with debris (initialise), then check latest image to compare
# if image has already been read
def check_saved_image(cam_name, img):
    res = os.listdir('./saved_images')

    # save an initial image to compare if dir is empty
    if len(res) == 0:
        save_image(cam_name, img)
        bts_img = convert_img_for_db(img)
        df_bts_img = pd.DataFrame([cam_name, bts_img])
        df_bts_img.rename(columns={0: 'Image'}, inplace=True)
        return df_bts_img

    else:
        list_of_files = glob.glob('saved_images/*.jpg')
        prev_img = max(list_of_files, key=os.path.getctime)
        result = is_similar(img, prev_img)

        if result:
            save_image(cam_name, img)
            bts_img = convert_img_for_db(img)
            df_bts_img = pd.DataFrame([cam_name, bts_img])
            df_bts_img.rename(columns={0: 'Image'}, inplace=True)
            return df_bts_img


# to calculate height of Bretby in the trough
# assumption, as Bretby height increases, there is coal underneath
def bret_loc_data(df_infer, cam_name, img, bb_results):
    try:
        # calc 0.39/0.1% of x/y
        # df = get_time_diff(df_time)
        if not df_infer.empty or not None:

            if bb_results:
                df = get_change_in_xy(df_infer)

                if not df.empty:

                    df['BretbyDebLoc_x'] = df.apply(lambda row: add_x(float(row['x'])), axis=1)
                    df['BretbyDebLoc_y'] = df.apply(lambda row: add_y(float(row['y'])), axis=1)
                    df['result_x'] = df.apply(lambda row: greater_x(float(row['x']), float(row['BretbyDebLoc_x'])),
                                              axis=1)
                    df['result_y'] = df.apply(lambda row: greater_y(float(row['y']), float(row['BretbyDebLoc_y'])),
                                              axis=1)

                    # if percentage of debris in Trough is >10% return True
                    df['DebrisResult'] = df[['PercentageTroughFull']].apply(
                        lambda x: True if x['PercentageTroughFull'] > pct_of_debris else False, axis=1)

                    # if bretby is high up the trough wall return True
                    df['BretbyResult'] = df.apply(lambda row: check_eqal((row['result_x']), (row['result_y'])), axis=1)

                    df = df[df['result_x'] == False]
                    df_pass = df.loc[:, df.columns.drop(
                        ['t0', 't1', 'x', 'y', 'diff_x', 'diff_y', 'result_x', 'result_y'])]

                    df_bts_img = check_saved_image(cam_name, img)
                    df_pass['Image'] = df_bts_img
                    df_pass = pd.concat([df_pass])

                    # result = df_pass['debris_result'].eq(False).all()
                    df_pass_true = df_pass[df_pass['DebrisResult'] == True]

                    # db manager gets results from csv file for db insert
                    df_pass_true.to_csv('temp_out.csv', index=False)
                    db_manager_controller()
    except Exception as e:
        print(e)

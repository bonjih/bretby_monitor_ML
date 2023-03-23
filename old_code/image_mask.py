__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import cv2 as cv
import imutils
import numpy as np
from imutils import contours, perspective


def get_hsv_flow():
    track1hsv = (np.array([121, 0, 16]), np.array([137, 255, 180]))
    track2hsv = (np.array([0, 0, 0]), np.array([120, 255, 255]))
    bretbyhsv = (np.array([20, 100, 100]), np.array([30, 255, 255]))
    return track1hsv, track2hsv, bretbyhsv


def get_box_coords(C, new_frame):
    """
    creates bounding boxes bounds on HSV parameters
    returens BB centre and coords array
    :param C:
    :param new_frame:
    :return:
    """
    box = cv.minAreaRect(C)
    M = cv.moments(C)
    box = cv.boxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
    box = perspective.order_points(box)

    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    cv.circle(new_frame, center, 5, (0, 0, 255), -1)
    cv.drawContours(new_frame, [box.astype("int")], -1, (0, 255, 255), 2)
    array_stacked = np.vstack([box])
    return array_stacked, center


def calcAreaPercent(tl, tr, bl, arr):
    box_area = (tr[0] - tl[0]) * (bl[1] - tl[1])
    total = (arr[0][1] - arr[3][1]) * (arr[1][0] - arr[0][0])
    percent = box_area / total

    return box_area, percent


def get_box_coords_debris(cnts, new_frame, arr):
    for C in cnts:
        if cv.contourArea(C) < 250:
            continue

        rect = cv.minAreaRect(C)
        box = cv.boxPoints(rect)
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        box_area, percent = calcAreaPercent(tl, tr, bl, arr)

        cv.drawContours(new_frame, [box.astype("int")], -1, (255, 255, 0), 2)
        cv.putText(new_frame, "Area: " + "{:.2f}".format(box_area * 0.36), (int(tr[0]), int(tr[1])),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (209, 80, 0, 255), 2)
        cv.putText(new_frame, "Percent: " + "{:.2f}".format(percent * 100), (int(tr[0]), int(tr[1]) + 40),
                   cv.FONT_HERSHEY_COMPLEX, 1.0, (209, 80, 0, 255), 2)
        return box_area, percent


def bitwise_convert(img, mask_arr):
    """
    :param img: video frame
    :param mask_arr: mask of bounding box from ML process
    :return:
    """
    if mask_arr is not None:
        zeroes = np.zeros_like(img)
        cv.fillPoly(zeroes, pts=[mask_arr], color=(255, 255, 255))
        mask = cv.bitwise_and(img, img, mask=zeroes)
        return mask
    else:
        return img


def make_mask(new_frame, np_array):
    track1hsv, track2hsv, bretbyhsv = get_hsv_flow()

    hsv = cv.cvtColor(new_frame, cv.COLOR_BGR2HSV)

    # process trough HSV
    track_hsv_img = cv.inRange(hsv, track2hsv[0], track2hsv[1])
    track_hsv_img = cv.erode(track_hsv_img, np.ones((3, 3), np.uint8), cv.BORDER_REFLECT)
    track_hsv_img = cv.dilate(track_hsv_img, np.ones((3, 3), np.uint8), cv.BORDER_REFLECT)

    mask_arr_t = bitwise_convert(track_hsv_img, np_array)
    cnts_trough = cv.findContours(mask_arr_t.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts_trough = imutils.grab_contours(cnts_trough)
    (cnts_trough, _) = contours.sort_contours(cnts_trough)

    # process bretby HSV
    bret_hsv_img = cv.inRange(hsv, bretbyhsv[0], bretbyhsv[1])
    mask_arr_b = bitwise_convert(bret_hsv_img, np_array)
    bret_cnts = cv.findContours(mask_arr_b.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bret_cnts = imutils.grab_contours(bret_cnts)

    (bret_cnts, _) = contours.sort_contours(bret_cnts)

    return cnts_trough, bret_cnts

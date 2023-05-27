__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import cv2
import numpy as np
import imutils
from imutils import perspective

lst = []


def make_pairwise(item):
    """
    gets previous and next contour len
    if the same len, keep, if not reject.
    reject means contours are different and box is moving
    """
    l = len(item)
    tups = list(zip([l], [l][:1] + [l][1:]))
    lst.append(tups)
    a = lst[0][0][1]
    b = lst[-1][0][0]
    return a == b


def draw_color_contours(frame, cnts):
    for c in cnts:
        area = cv2.contourArea(c)
        
        if 1000 <= area <= 6000:
            cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)
            rect = cv2.minAreaRect(c)
            box_area = rect[1][0] * rect[1][1]
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if 20.0 < rect[1][0] and rect[1][1] < 100:  # filter irrelevant out boxes
                box = cv2.boxPoints(rect)
                box = perspective.order_points(box)
                (tl, tr, br, bl) = box

                result = make_pairwise(c)

                if result:
                    percent = round((area / box_area) * 100, 2)
                    cv2.drawContours(frame, [box.astype("int")], -1, (255, 255, 0), 2)
                    cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                    cv2.putText(frame, "%" + "{}".format(percent), (int(tr[0]), int(tr[1]) + 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
                    return percent, result

    return None, None


def make_hsv(arr):
    arr = cv2.blur(arr, ksize=(15, 15))
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)

    trackhsv = (np.array([121, 0, 16]), np.array([137, 255, 180]))
    bretbyhsv = (np.array([20, 100, 100]), np.array([30, 255, 255]))

    track_hsv = cv2.inRange(hsv, trackhsv[0], trackhsv[1])
    bretby_hsv = cv2.inRange(hsv, bretbyhsv[0], bretbyhsv[1])
    return track_hsv, bretby_hsv


def get_box_coords(C, new_frame):
    box = cv2.minAreaRect(C)
    M = cv2.moments(C)
    box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = perspective.order_points(box)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    cv2.circle(new_frame, center, 5, (0, 0, 255), -1)
    cv2.drawContours(new_frame, [box.astype("int")], -1, (0, 255, 255), 2)
    array_stacked = np.vstack([box])
    return array_stacked, center


def bitwise_convert(img, mask_arr):
    if mask_arr is not None:
        zeroes = np.zeros_like(img)
        cv2.fillPoly(zeroes, pts=[mask_arr], color=(255, 255, 255))
        mask = cv2.bitwise_and(img, img, mask=zeroes)
        return mask
    else:
        return img


def make_roi(hsv_img, np_array):
    masked = bitwise_convert(hsv_img, np_array)
    cnts = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts


def make_mask(frame, np_array):
    kernel = np.ones((3, 3), np.uint8)
    trough_hsv, bretby_hsv = make_hsv(frame)
    trough_hsv = cv2.dilate(trough_hsv, kernel, cv2.BORDER_REFLECT)
    trough_hsv = cv2.erode(trough_hsv, kernel, cv2.BORDER_REFLECT)
    trough_hsv = cv2.morphologyEx(trough_hsv, cv2.MORPH_OPEN, kernel)
    trough_hsv = cv2.morphologyEx(trough_hsv, cv2.MORPH_CLOSE, kernel)
    cnts_trough = make_roi(trough_hsv, np_array)
    cnts_bretby = make_roi(bretby_hsv, np_array)

    if len(cnts_bretby) == 0:
        cnts_bretby = cnts_trough
    return cnts_trough, cnts_bretby


def find_contours(frame, np_array):
    no_blur = frame
    cnts_trough, bretby_hsv = make_mask(frame, np_array)
    percent, result = draw_color_contours(no_blur, cnts_trough)
    return cnts_trough, bretby_hsv, percent, result

__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import cv2 as cv
import numpy as np
from datetime import datetime

import data_processing
from data_format import format_df, format_data
from debris_detect import find_contours
from debris_detect import get_box_coords
from model.utils.transforms import get_bb_coords

# PARAMETERS------------------------------------------------------------------
previewWindow = True

# visualisation parameters
numPts = 1  # max number of points to track
trailLength = 2  # how many frames to keep a fading trail behind a tracked point to show motion
trailThickness = 4  # thickness of the trail to draw behind the target
trailFade = 4  # the intensity at which the trail fades
pointSize = 4  # pixel radius of the circle to draw over tracked points

# params for Shi-Tomasi corner detection
shitomasi_params = {
    "qualityLevel": 0.3,  # minimal accepted quality of image corners
    "minDistance": 7,  # minimum possible Euclidean distance between the returned corners
    "blockSize": 7  # size of an average block for computing a derivative cov matrix over each pixel neighbourhood
}

# params for Lucas-Kanade optical flow
LK_params = {
    "winSize": (9, 9),
    "maxLevel": 2,
    "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
}


def get_bretby_pts(cnts, new_frame):
    if cnts is not None:
        for C_flow in cnts:
            if cv.contourArea(C_flow) < 1:
                continue

            coords_array, center = get_box_coords(C_flow, new_frame)

            return coords_array, center


# SETUP -----------------------------------------------------------------------

# generate random colors
color = np.random.randint(0, 255, (100, 3))


def create_crosshairs(old_frame, box_coords):
    crosshairmask = np.zeros(old_frame.shape[:2], dtype="uint8")
    cv.rectangle(crosshairmask, (int(box_coords[3][0]), int(box_coords[3][1])), (int(box_coords[1][0]),
                                                                                 int(box_coords[1][1])), 255, -1)
    box_xy = (int(box_coords[3][0]), int(box_coords[3][1]))
    return crosshairmask, box_xy


trail_history = [[[(0, 0), (0, 0)] for _ in range(trailLength)] for _ in range(numPts)]


# PROCESS VIDEO ---------------------------------------------------------------
def bret_flow(new_frame, old_gray, old_frame, cam_name, x, y):
    global crosshair_mask
    np_array = get_bb_coords(x, y)

    results = []
    old_points = None
    new_points = None
    st = [[1]]
    pcts = 0

    try:
        cnts_trough, bret_cnts, pcts, result = find_contours(new_frame, np_array)

        results.append(result)

        coords_array, center = get_bretby_pts(bret_cnts, new_frame)

        crosshair_mask, box_xy = create_crosshairs(old_frame, coords_array)

        # get features from first frame
        old_points = cv.goodFeaturesToTrack(old_gray, maxCorners=numPts, mask=crosshair_mask, **shitomasi_params)

        # convert to grayscale
        new_frame_gray = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)

        # due to the narrow window of old_points, null values exists
        if old_points is None:
            old_points = np.array([[[box_xy[0], box_xy[1]]]])

        # https://stackoverflow.com/questions/34540181/opencv-optical-flow-assertion ..... np.float32(old_points)
        # calculate optical flow
        new_points, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_frame_gray, np.float32(old_points), None,
                                                      **LK_params)
    except Exception as e:
        pass  # TODO make if statements when no bretby to track
        # print(e, 'bretflow_newPoints -', datetime.now())

    # select good points
    if old_points is not None and new_points is not None:
        good_new = new_points[st == 1]
        good_old = old_points[st == 1]

        # create trail mask to add to image
        trailMask = np.zeros_like(old_frame)

        # calculate motion lines and points
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # flatten coords
            a, b = new.ravel()
            c, d = old.ravel()

            # list of the prev and current points converted to int
            line_pts = [(int(a), int(b)), (int(c), int(d))]

            # add points to the trail history
            trail_history[i].insert(0, line_pts)

            # get color for this point
            pointColor = color[i].tolist()

            # add trail lines
            for j in range(len(trail_history[i])):
                trailColor = [int(pointColor[0] - (trailFade * j)), int(pointColor[1] - (trailFade * j)),
                              int(pointColor[2] - (trailFade * j))]  # fading colors
                trailMask = cv.line(trailMask, trail_history[i][j][0], trail_history[i][j][1], trailColor,
                                    thickness=trailThickness, lineType=cv.LINE_AA)

            tup_1 = trail_history[i][0][0]
            tup_2 = trail_history[i][0][-1]

            res = all(map(lambda p, v: p > v, tup_1, tup_2))

            # get rid of the trail segment
            trail_history[i].pop()
            loc = trail_history[i][0][0]

            # add circle over the point
            orig_frame = cv.circle(new_frame, trail_history[i][0][0], pointSize, color[i].tolist(), -1)
            cv.putText(orig_frame, 'Bretby Tracker', (loc[0], loc[1]), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12),
                       2)
            if pcts is None:
                pcts = 0

            format_data(i, cam_name, res, tup_1, tup_1, pcts, trail_history)

        # add trail to frame
        img = cv.add(new_frame, trailMask)

        # update previous frame and previous points
        old_gray = new_frame_gray.copy()
        old_points = good_new.reshape(-1, 1, 2)

        # if old_points < numPts, get new points
        if (numPts - len(old_points)) > 0:
            old_points = cv.goodFeaturesToTrack(old_gray, maxCorners=numPts, mask=crosshair_mask,
                                                **shitomasi_params)
    return new_frame, results


def bret_flow_run(orig_frame, old_gray, old_frame, cam_name, x, y):

    try:
        images, results = bret_flow(orig_frame, old_gray, old_frame, cam_name, x, y)

        df = format_df()

        if df is not None:
            data_processing.bret_loc_data(df, cam_name, images, results[-1])
        else:
            pass
    except Exception as e:
        pass
        #print(e, 'bretflow_run', datetime.now())

import cv2 as cv
import numpy as np

import data_processing
from data_format import format_df, format_data
from image_mask import make_mask, get_box_coords_percent, get_box_coords
from model.utils.transforms import get_bb_coords

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

# SETUP -----------------------------------------------------------------------

# generate random colors
color = np.random.randint(0, 255, (100, 3))


def get_deb_pcts(new_frame, np_array):
    try:
        cnts, _ = make_mask(new_frame, np_array)

        for C in cnts:
            if cv.contourArea(C) < 250:
                continue

            box_area, percent = get_box_coords_percent(C, new_frame, np_array)

            return percent
    except:
        pass


def get_brtby_pts(new_frame, np_array):
    try:

        _, bret_cnts = make_mask(new_frame, np_array)

        for C_flow in bret_cnts:
            if cv.contourArea(C_flow) < 1:
                continue

            coords_array, center = get_box_coords(C_flow, new_frame)
            return coords_array, center
    except:
        pass


def create_crosshairs(old_frame, box_coords):
    crosshairmask = np.zeros(old_frame.shape[:2], dtype="uint8")
    cv.rectangle(crosshairmask, (int(box_coords[3][0]), int(box_coords[3][1])),
                 (int(box_coords[1][0]), int(box_coords[1][1])), 255, -1)
    return crosshairmask, int(box_coords[3][0]), int(box_coords[3][1])


def process_flow(orig_frame, old_frame, old_points, crosshair_mask, trail_history, old_gray, cam_name, x1, y1):
    # PROCESS VIDEO ---------------------------------------------------------------

    pcts = 0

    np_array = get_bb_coords(x1, y1)

    try:
        pcts = get_deb_pcts(orig_frame, np_array)

        if pcts is None:
            pcts = 0

    except:
        pass

    # LK needs grayscale
    new_frame_gray = cv.cvtColor(orig_frame, cv.COLOR_BGR2GRAY)

    # due to the narrow window of old_points, null values exists
    # https://stackoverflow.com/questions/34540181/opencv-optical-flow-assertion ..... np.float32(old_points)
    if old_points is None:
        old_points = np.array([[[x1, y1]]])

    # calculate optical flow
    new_points, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_frame_gray, old_points, None, **LK_params)

    # select good points
    good_new = new_points[st == 1]
    good_old = old_points[st == 1]

    # create trail mask to add to image
    trail_mask = np.zeros_like(old_frame)

    # calculate motion lines and points
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # flatten coords
        a, b = new.ravel()
        c, d = old.ravel()

        # list of the prev and current points converted to int
        linepts = [(int(a), int(b)), (int(c), int(d))]

        # add points to the trail history
        trail_history[i].insert(0, linepts)

        # get color for this point
        point_color = color[i].tolist()

        # add trail lines
        for j in range(len(trail_history[i])):
            trail_color = [int(point_color[0] - (trailFade * j)), int(point_color[1] - (trailFade * j)),
                           int(point_color[2] - (trailFade * j))]  # fading colors
            trail_mask = cv.line(trail_mask, trail_history[i][j][0], trail_history[i][j][1], trail_color,
                                 thickness=trailThickness, lineType=cv.LINE_AA)

        tup_1 = trail_history[i][0][0]
        tup_2 = trail_history[i][0][-1]

        res = all(map(lambda x, y: x > y, tup_1, tup_2))

        # get rid of the trail segment
        trail_history[i].pop()
        loc = trail_history[i][0][0]

        # add circle over the point
        orig_frame = cv.circle(orig_frame, trail_history[i][0][0], pointSize, color[i].tolist(), -1)
        cv.putText(orig_frame, 'Bretby Tracker', (loc[0], loc[1]), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        format_data(i, cam_name, res, tup_1, tup_1, pcts, trail_history)

    # add trail to frame
    img = cv.add(orig_frame, trail_mask)

    # update previous frame and previous points
    old_gray = new_frame_gray.copy()
    old_points = good_new.reshape(-1, 1, 2)

    # if old_points < numPts, get new points
    if (numPts - len(old_points)) > 0:
        old_points = cv.goodFeaturesToTrack(old_gray, maxCorners=numPts, mask=crosshair_mask, **shitomasi_params)

    return orig_frame


def bret_flow(orig_frame, old_frame, old_gray, cam_name, x, y):
    np_array = get_bb_coords(x, y)

    try:
        coords_array, center = get_brtby_pts(orig_frame, np_array)
        create_crosshairs(old_frame, coords_array)
        crosshairmask, x3, y3 = create_crosshairs(old_frame, coords_array)

        # create masks for drawing purposes
        trail_history = [[[(0, 0), (0, 0)] for _ in range(trailLength)] for _ in range(numPts)]

        # get features from first frame
        old_points = cv.goodFeaturesToTrack(old_gray, maxCorners=numPts, mask=crosshairmask, **shitomasi_params)

        images = process_flow(orig_frame, old_frame, old_points, crosshairmask, trail_history, old_gray, cam_name, x, y)

        df = format_df()

        if df is not None:
            data_processing.bret_loc_data(df, cam_name, images)
        else:
            pass

    except:
        pass

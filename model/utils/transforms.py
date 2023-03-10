__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import numpy as np
import cv2

from torchvision import transforms as transforms


def resize(im, img_size=640, square=False):
    # Aspect ratio resize
    if square:
        im = cv2.resize(im, (img_size, img_size))
    else:
        h0, w0 = im.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
    return im


# makes the new bb coords for annotation:
def get_bb_coords_new_bb(tl0, br0):
    """
    :param tl0: box x
    :param br0: box y
    :return: np array of bounding box
    """
    bl = (tl0[0], br0[1])
    br = br0
    tr = (br0[0], tl0[1])
    new_tl = tl0[0] + 50
    tl = (new_tl, tl0[1])
    arr = np.array([bl, br, tr, tl])
    return arr


# makes the bb coords for array input to find_mask as:
# bl, br, tl, tr
def get_bb_coords(tl0, br0):
    """
    :param tl0: box x
    :param br0: box y
    :return: np array of bounding box
    """
    try:
        if len(tl0) and len(br0) != 0:
            bl = (tl0[0][0], br0[0][1])
            tr = (br0[0][0], tl0[0][1])
            arr = np.array([bl, br0[0], tr, tl0[0]])
            return arr
    except:
        pass


def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)

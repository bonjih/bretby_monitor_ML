__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import cv2 as cv
import numpy as np
from torchvision import transforms as transforms


def get_bb_coords_new_bb(tl, br):
    """
    Calculates the new bounding box coordinates for annotation.
    :param tl: Top-left corner of the box (x, y)
    :param br: Bottom-right corner of the box (x, y)
    :return: Numpy array of new bounding box coordinates
    """
    bl = (tl[0], br[1])
    br = br
    tr = (br[0], tl[1])
    new_tl = tl[0] + 50
    new_tl_coord = (new_tl, tl[1])
    arr = np.array([bl, br, tr, new_tl_coord])
    return arr


def get_bb_coords(tl, br):
    """
    Calculates the bounding box coordinates as (bl, br, tl, tr) for array input.
    :param tl: Top-left corner of the box (x, y)
    :param br: Bottom-right corner of the box (x, y)
    :return: Numpy array of bounding box coordinates
    """
    try:
        if len(tl) and len(br) != 0:
            bl = (tl[0][0], br[0][1])
            tr = (br[0][0], tl[0][1])
            arr = np.array([bl, br[0], tr, tl[0]])
            return arr
    except:
        pass


def infer_transforms(image):
    """
    Applies torchvision image transforms to the image.
    :param image: Input image
    :return: Transformed image
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)


def is_similar(current_image, prev_image_path):
    """
    Checks if two images are similar by comparing their shapes and bitwise XOR.
    :param current_image: Current image as a NumPy array
    :param prev_image_path: Path to the previous image
    :return: True if the images are similar, False otherwise
    """
    previous_image = cv.imread(prev_image_path)
    return current_image.shape == previous_image.shape and not np.bitwise_xor(current_image, previous_image).any()


def convert_img_for_db(image):
    """
    Converts the image data to binary format for database insertion.
    :param image: Image data as a NumPy array
    :return: Image data in bytes format
    """
    img_encode = cv.imencode('.jpg', image)[1]
    data_encode = np.array(img_encode)
    bts = data_encode.tobytes()
    return bts

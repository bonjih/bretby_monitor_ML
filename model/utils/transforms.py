__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import numpy as np

from torchvision import transforms as transforms


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


# a check to see if to images are the same before saving to file/db
def is_similar(image1, image2):
    return image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any())

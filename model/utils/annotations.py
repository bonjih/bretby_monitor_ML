__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import numpy as np
import torch
from torchvision.ops import nms
import cv2

from model.utils.transforms import get_bb_coords_new_bb

no_labels = False


def inference_annotations(
        outputs,
        detection_threshold,
        classes,
        colors,
        orig_image,
        image
):
    height, width, _ = orig_image.shape
    boxes = outputs[0]['boxes'].data.numpy()

    scores = outputs[0]['scores'].data.numpy()

    # Filter out boxes according to 'detection_threshold'.
    boxes = boxes[scores >= detection_threshold].astype(np.int32)

    #
    # boxes = torch.from_numpy(np.array(boxes))
    #
    # scores = torch.from_numpy(np.array(scores))
    #
    # nms(boxes=boxes, scores=scores, iou_threshold=0.2)
    #
    # boxes = boxes.numpy()
    # scores = scores.numpy()
    draw_boxes = boxes.copy()
    # Get all the predicted class names.
    pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]

    lw = max(round(sum(orig_image.shape) / 2 * 0.0025), 2)  # Line width.

    tl = []
    br = []

    # Draw the bounding boxes and write the class name on top of it.
    for j, box in enumerate(draw_boxes):
        p1 = (int(box[0] / image.shape[1] * width), int(box[1] / image.shape[0] * height))
        p2 = (int(box[2] / image.shape[1] * width), int(box[3] / image.shape[0] * height))

        # ensure len is within img bounds
        off_set = 50
        new_len = int(p2[1] - off_set)
        p2 = (p2[0], new_len)

        # fit bb to perspective of trough
        arr = get_bb_coords_new_bb(p1, p2)

        class_name = pred_classes[j]
        color = colors[classes.index(class_name)]

        # draws the bounding box as contours over a square
        # cv2.drawContours(orig_image, [arr], 0, color=color, thickness=lw,
        #                  lineType=cv2.LINE_AA)

        cv2.rectangle(
            orig_image,
            p1, p2,
            color=color,
            thickness=lw,
            lineType=cv2.LINE_AA
        )

        tl.append(p1)
        br.append(p2)

        if not no_labels:
            # For filled rectangle.
            final_label = class_name + ' ' + str(round(scores[j], 2))
            w, h = cv2.getTextSize(
                final_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=lw / 3.5,
                thickness=1
            )[0]  # text width, height

            w = int(w - (0.15 * w))
            outside = p1[1] - h >= 1
            p2 = p1[0] + w, p1[1] - h - 1 if outside else p1[1] + h + 1
            p1_b = p1[0] + off_set  # score txt offset
            p1_b = p1_b, p1[1]

            cv2.rectangle(
                orig_image,
                p1_b,
                p2,
                color=color,
                thickness=-1,
                lineType=cv2.LINE_AA
            )

            cv2.putText(
                orig_image,
                final_label,
                (p1[0] + off_set, p2[1] + 9),  # box txt location
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=lw / 8.8,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA
            )
    return orig_image, tl, br


def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=1,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
):
    offset = (5, 5)
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_start = tuple(x - y for x, y in zip(pos, offset))
    rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
    cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
    cv2.putText(
        img,
        text,
        (x, int(y + text_h + font_scale - 1)),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )
    return img


def annotate_fps(orig_image, fps_text):
    draw_text(
        orig_image,
        f"FPS: {fps_text:0.1f}",
        pos=(20, 20),
        font_scale=1.0,
        text_color=(204, 85, 17),
        text_color_bg=(255, 255, 255),
        font_thickness=2,
    )
    return orig_image

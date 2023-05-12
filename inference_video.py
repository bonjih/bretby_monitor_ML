__author__ = ""
__email__ = ""
__phone__ = ""
__license__ = "xxx"
__version__ = "1.0.0"
__maintainer__ = ""
__status__ = "Dev"

import imutils
import numpy as np
import cv2
import torch
import time
import yaml
from datetime import datetime

import global_conf_variables
from model.models.create_fasterrcnn_model import create_model
from model.utils.annotations import inference_annotations, annotate_fps
from model.utils.transforms import infer_transforms
from bretby_flow import bret_flow_run
from utils.save_vid import vid_save

values = global_conf_variables.get_values()

# PARAMETERS------------------------------------------------------------------
timer = values[1]
save_vid = values[2]
preview_window = values[3]

img_size = 960
configs = 'C:\\bretby_monitor_ML\\model\\data_configs\\custom_data.yaml'
weights = 'C:\\bretby_monitor_ML\\model\\last_model_state_15032023.pth'
threshold = 0.90  # detection threshold - any detection having score below this will be discarded.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = 'fasterrcnn_resnet50_fpn_v2'


def read_return_video(video_path, cam_name):
    try:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

        result = cap.isOpened()
        if result:
            # get the first frame
            _, old_frame = cap.read()

            if img_size is not None:
                old_frame = imutils.resize(old_frame, width=img_size)
            else:
                old_frame = old_frame

            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

            frame_width = old_frame.shape[1]
            frame_height = old_frame.shape[0]
            assert (frame_width != 0 and frame_height != 0), 'Please check video path...'
            return cap, frame_width, frame_height, old_frame, old_gray

        else:
            from stream_manager import probe_stream
            probe_stream(video_path, cam_name)

    except Exception as e:
        print(e, 'Infer_read_vid', datetime.now())


def main(cam_name, source):
    # For same annotation colors each time.
    np.random.seed(42)

    # Load the data configurations.
    with open(configs) as file:
        data_configs = yaml.safe_load(file)

    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']

    DEVICE = device

    # Load trained weights
    if weights is not None:
        checkpoint = torch.load(weights, map_location=DEVICE)
        # If config file is not given, load from model dictionary.
        if data_configs is None:
            data_configs = True
            NUM_CLASSES = checkpoint['config']['NC']
            CLASSES = checkpoint['config']['CLASSES']
        try:
            print('Building from model name arguments...')
            build_model = create_model[models]
        except:
            build_model = create_model[checkpoint['model_name']]

        model = build_model(num_classes=NUM_CLASSES, coco_model=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE).eval()

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    detection_threshold = threshold

    cap, frame_width, frame_height, old_frame, old_gray = read_return_video(source, cam_name)

    # Save Video
    fps = cap.get(cv2.CAP_PROP_FPS)

    if save_vid:
        video_out = vid_save(fps, frame_width, frame_height, cam_name)

    if img_size is not None:
        RESIZE_TO = img_size
    else:
        RESIZE_TO = frame_width

    frame_count = 0  # To count total frames.
    total_fps = 0  # To get the final frames per second.

    t0 = time.time()

    while True:

        # capture each frame of the video
        ret, frame = cap.read()

        t1 = time.time()
        time_out = t1 - t0

        if time_out > timer:
            break

        if ret:
            frame = imutils.resize(frame, width=RESIZE_TO)
            orig_frame = frame.copy()
            image = frame.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = infer_transforms(image)
            # Add batch dimension.
            image = torch.unsqueeze(image, 0)

            # Get the start time.
            start_time = time.time()

            with torch.no_grad():
                # Get predictions for the current frame.
                outputs = model(image.to(DEVICE))
            forward_end_time = time.time()

            forward_pass_time = forward_end_time - start_time

            # Get the current fps.
            fps = 1 / forward_pass_time
            # Add 'fps' to 'total_fps'
            total_fps += fps
            # Increment frame count.
            frame_count += 1

            # Load all detection to CPU for further operations.
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            # only if there are detected boxes.
            if len(outputs[0]['boxes']) != 0:
                frame, x, y = inference_annotations(outputs, detection_threshold, CLASSES, COLORS, orig_frame, frame)

                if len(x) != 0 and len(x) != 0:
                    bret_flow_run(orig_frame, old_gray, old_frame, cam_name, x, y)
                    pass
            else:

                frame = orig_frame

            frame = annotate_fps(frame, fps)
            #
            # final_end_time = time.time()
            # forward_and_annot_time = final_end_time - start_time

            if save_vid:
                video_out.write(frame)

            if preview_window:
                # cv2.imshow(cam_name, frame)
                pass
                # Press `q` to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


def inf_run(cam_name, camID):
    main(cam_name, camID)

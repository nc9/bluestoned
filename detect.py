"""

"""
import os
import sys
import time
import logging
import cv2
import numpy as np

logging.basicConfig(
    level=logging.DEBUG
)

# this is the upper and lower blue color bounds we are searching for
# opencv HSV values differ from most image editing programs hence the calcs
BLUE_LOWER = np.array([232 / 2, 0.81 * 255, 0.75 * 255])
BLUE_UPPER = np.array([240 / 2, 1 * 255, 1 * 255])


def analyze_image(img_file, show_result=False, save_result=False, save_path=None):
    if not os.path.isfile(img_file):
        raise Exception("Not an image: {}".format(img_file))

    frame = cv2.imread(img_file)
    mask, mask_count = _get_mask_for_frame(frame)

    if mask_count:
        logging.debug(
            "Found screen mask %d in image %s",
            mask_count,
            img_file
        )

        frame = _draw_bounding_boxes(frame, mask, mask_count)

    result_filename = _get_result_filename(img_file)
    result_directory = os.getcwd() if not save_path else None

    if show_result:
        cv2.imshow(result_filename, frame)

        # keep the image open until the user hits q
        while True:
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    if save_result:
        result_filepath = os.path.join(
            result_directory,
            result_filename,
        )

        logging.debug(
            "Saving result file to %s",
            result_filepath
        )

        cv2.imwrite(
            result_filepath,
            frame
        )


    cv2.destroyAllWindows()

def analyze_video(video_file, max_only=True, output_video=None):
    if not os.path.isfile(video_file):
        raise Exception("Invalid video file: {}".format(video_file))

    fps_read_rate = 30
    start_time = time.time()

    cap = cv2.VideoCapture(video_file)

    time.sleep(3)
    frame_count = 0
    frames_processed = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    logging.info(
        "Opened video (%ffps %dx%d)",
        fps,
        frame_width,
        frame_height
    )


    out = None
    max_mask_count = 0
    max_mask_time = 0
    max_mask_frame = None

    if output_video:
        out = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc('M','J','P','G'),
            10,
            (frame_width, frame_height)
        )

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            logging.info("End of video")
            break

        frame_count += 1
        time_cur = time.time()

        if (int(fps) == 60 and (frame_count % 2 == 0)):
            continue

        if (int(time_cur - start_time)) > fps_read_rate:
            start_time = time_cur
            continue

        frames_processed += 1

        mask, mask_count = _get_mask_for_frame(frame)

        if not mask_count and max_mask_count:
            logging.info(
                "Frame found at %s",
                max_mask_time
            )
            cv2.imshow('image', max_mask_frame)

        if not mask_count:
            max_mask_count = 0

        if mask_count:
            logging.debug(
                "Found in frame %d at %s with mask count %d of max %d",
                frame_count,
                _time_conver_ms_to_timestring(cap.get(cv2.CAP_PROP_POS_MSEC)),
                mask_count,
                max_mask_count
            )

            frame = _draw_bounding_boxes(frame, mask, mask_count)

            # save the peak mask_count frame for this detection
            if mask_count > max_mask_count:
                max_mask_count = mask_count
                max_mask_frame = frame
                max_mask_time = cap.get(cv2.CAP_PROP_POS_MSEC)

        if out:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if out:
        out.release()

    cap.release()
    cv2.destroyAllWindows()

    logging.info("Processed %d of %d frames", frames_processed, frame_count)

def _get_mask_for_frame(frame):
    " for a frame get the mask pixel count "
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsv = cv2.medianBlur(frame_hsv, 5)

    mask = cv2.inRange(frame_hsv, BLUE_LOWER, BLUE_UPPER)
    mask_count = cv2.countNonZero(mask)
    return mask, mask_count

def _draw_bounding_boxes(frame, mask, mask_count=0):
    " draw the bounding boxes on the frame based on the mask "
    # find the countour from the bitmask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # draw the rectangle around the contour
    x, y, w, h = cv2.boundingRect(contours_sorted[0])

    if mask_count < 3:
        # cv2.imshow('Frame', frame)
        rec_color = (255, 0, 0)
        rec_size = 5
    elif mask_count >= 3 and mask_count <= 30:
        rec_color = (0, 255, 0)
        rec_size = 5
    else:
        rec_color = (0, 0, 255)
        rec_size = 5

    cv2.rectangle(frame, (x - 30, y - 30), (x + w + 60, y + h + 60), rec_color, rec_size)
    return frame

def _get_result_filename(filename):
    _components = os.path.splitext(os.path.basename(filename))
    return "{}-result{}".format(_components[0], _components[1])

def _time_conver_ms_to_timestring(millis):
    " convert millisecond time delta as a float into a string describing time in HH:mm:ss "
    millis = float(millis)
    seconds = (millis / 1000) % 60
    minutes = (millis / (1000 * 60)) % 60
    hours = (millis / (1000 * 60 * 60)) % 24
    return "{0:02d}:{1:02d}.{2:02d}".format(int(hours), int(minutes), int(seconds))

def analyze_image_dir(_dir):
    """ walk a directory listing and read and process all found images """
    if not os.path.isdir(_dir):
        raise Exception("Invalid directory: {}".format(_dir))

    for file_name in os.listdir(_dir):
        analyze_image("{}/{}".format(_dir, file_name))

if __name__ == "__main__":
    try:
        analyze_image("images/rda-907.jpeg", save_result=True)
        # listdir("ex02")
        # main()
        # analyze_video("videos/rda-flash.mp4")
    except KeyboardInterrupt:
        logging.error("Interrupted")
    except Exception as ex:
        logging.exception(ex)

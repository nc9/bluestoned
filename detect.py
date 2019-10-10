"""
    ~~ bluestoned ~~

    detect chroma keys in video and image files

    (c) 2019 Nik Cubrilovic <git@nikcub.me>

"""
import os
import sys
import time
import logging
import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG
)

# this is the upper and lower blue color bounds we are searching for
# opencv HSV values differ from most image editing programs hence the calcs
BLUE_LOWER = np.array([232 / 2, 0.81 * 255, 0.75 * 255])
BLUE_UPPER = np.array([240 / 2, 1 * 255, 1 * 255])

VALID_VIDEO_EXT = set([
    ".mp4",
    ".mkv",
    ".avi",
    # ".webm"
])

VALID_IMAGE_EXT = set([
    ".jpeg",
    ".jpg",
    ".png",
])

def analyze_image(img_file, show_detections=False, save_detections=False, save_detections_path=None):
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

        _draw_bounding_boxes(frame, mask, mask_count)

    result_directory = "outputs" if not save_detections_path else save_detections_path
    result_directory = os.path.realpath(os.path.expanduser(result_directory))

    if not os.path.isdir(result_directory):
        os.mkdir(result_directory)

    result_filename = _get_result_filename(img_file)

    _draw_timestamp(frame, " {}".format(mask_count))

    if show_detections:
        cv2.imshow(result_filename, frame)

        # keep the image open until the user hits q
        while True:
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    if save_detections:
        result_filepath = os.path.join(
            result_directory,
            result_filename,
        )

        logging.info(
            "Saving result file to %s",
            result_filepath
        )

        cv2.imwrite(
            result_filepath,
            frame
        )


    cv2.destroyAllWindows()

def analyze_video(video_file, max_only=True, output_video=None, show_detections=False, save_detections=False, save_detections_path=None):
    """
    analyze a video to spot bluescreens/chroma keys and write the result


    """
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
    frame_total_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(
        "Opened %s (%ffps %dx%d) with %s total frames",
        video_file,
        fps,
        frame_width,
        frame_height,
        frame_total_length
    )

    out = None
    max_mask_count = 0
    max_mask_time = 0
    max_mask_frame = None
    detections = 0

    result_directory = "outputs" if not save_detections_path else save_detections_path
    result_directory = os.path.realpath(os.path.expanduser(result_directory))

    if not os.path.isdir(result_directory):
        os.mkdir(result_directory)

    if output_video:
        output_video_extension = "mp4"

        output_video_file = "{}-output.{}".format(
            os.path.splitext(video_file)[0],
            output_video_extension
        )

        output_video_path = os.path.join(
            result_directory,
            output_video_file
        )

        out = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), # @TODO this is a shit format find something better
            10,
            (frame_width, frame_height)
        )

        logging.info(
            "Saving output video to %s",
            output_video_path
        )

    pbar = tqdm(total=frame_total_length, unit="frames", )

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            logging.info("End of video")
            pbar.update(frame_total_length)
            break

        frame_count += 1

        # this is a bit of a hack but we know the source is < 60fps
        if (int(fps) == 60 and (frame_count % 2 == 0)):
            pbar.update(1)
            continue

        # if (int(time_cur - start_time)) > fps_read_rate:
        #     start_time = time_cur
        #     continue

        frames_processed += 1
        pbar.update(1)

        mask, mask_count = _get_mask_for_frame(frame)

        if not mask_count and max_mask_count:
            logging.info(
                "Mask count %d found in %s at %s",
                max_mask_count,
                video_file,
                _time_conver_ms_to_timestring(max_mask_time)
            )

            result_filename = _get_result_filename(video_file, "jpeg", detections)

            _draw_timestamp(max_mask_frame, "{} ({})".format(
                _time_conver_ms_to_timestring(max_mask_time),
                max_mask_count
            ))

            if save_detections:
                result_filepath = os.path.join(
                    result_directory,
                    result_filename,
                )

                logging.info(
                    "Saving result file to %s",
                    result_filepath
                )

                cv2.imwrite(
                    result_filepath,
                    max_mask_frame
                )

            if show_detections:
                cv2.imshow(result_filename, max_mask_frame)

            detections += 1

        if not mask_count:
            max_mask_count = 0

        if mask_count:
            logging.debug(
                "Mask count %d found in %s at %s (%d)",
                mask_count,
                video_file,
                _time_conver_ms_to_timestring(cap.get(cv2.CAP_PROP_POS_MSEC)),
                frame_count
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

    time_finish = time.time()

    logging.info(
        "Processed %d of %d frames in %s and found %d key frames in %s",
        frames_processed,
        frame_count,
        video_file,
        detections,
        _get_mask_for_frame(time_finish - start_time)
    )

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

def _draw_timestamp(frame, timestamp, height_from_bottom = 30):
    _height, _width, _ = frame.shape
    _font = cv2.FONT_HERSHEY_PLAIN
    _color = (255, 255, 255)

    text_y = _height - height_from_bottom

    try:
        cv2.putText(frame, str(timestamp), (50, text_y), _font, 4, _color, 3, cv2.LINE_AA)
        return True
    except Exception as ex:
        logging.error("Could not write timestamp to frame: %s", str(ex))
        return False


def _get_result_filename(filename, extension=None, index=None):
    _components = os.path.splitext(os.path.basename(filename))

    file_name = _components[0]
    file_extension = extension or _components[1][1:]
    index_format = ""

    if type(index) is int:
        index_format = "-{0:05d}".format(index)

    return "{}-result{}.{}".format(file_name, index_format, file_extension)

def _time_conver_ms_to_timestring(millis):
    " convert millisecond time delta as a float into a string describing time in HH:mm:ss "
    millis = float(millis)
    seconds = (millis / 1000) % 60
    minutes = (millis / (1000 * 60)) % 60
    hours = (millis / (1000 * 60 * 60)) % 24
    return "{0:02d}:{1:02d}.{2:02d}".format(int(hours), int(minutes), int(seconds))

def analyze_dir(_dir, **kwargs):
    """ walk a directory and analyze videos """
    if not os.path.isdir(_dir):
        raise Exception("Invalid directory: {}".format(_dir))

    for file_name in os.listdir(_dir):
        _file_ex = os.path.splitext(file_name)[1]

        if _file_ex in VALID_VIDEO_EXT:
            analyze_video(os.path.join(_dir, file_name), **kwargs)

        if _file_ex in VALID_IMAGE_EXT:
            analyze_image(os.path.join(_dir, file_name), **kwargs)

if __name__ == "__main__":
    try:
        # analyze_image("images/rda-907.jpeg", save_result=True)
        # listdir("ex02")
        # main()
        analyze_dir("videos", save_detections=True)
        # analyze_video("videos/rda-flash.mp4", save_detections=True)
    except KeyboardInterrupt:
        logging.error("Interrupted")
    except Exception as ex:
        logging.error(str(ex))
        logging.exception(ex)

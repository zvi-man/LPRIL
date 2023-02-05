import os.path
from typing import Tuple

import pandas as pd
import cv2
from tqdm import tqdm

# Constants
DEFAULT_COL_TO_PRINT = "track_id"
TOP_LEFT_X_COL = "top_l"
TOP_LEFT_Y_COL = "top_r"
WIDTH_COL = "width"
HEIGHT_COL = "height"
FRAME_COL = "frame"
GREEN = (0, 255, 0)
BBOX_COLOR = GREEN
BBOX_LINE_WIDTH = 2
TEXT_LINE_WIDTH = 2
TEXT_SIZE = 0.5
TEXT_COLOR = GREEN


def get_video_info(vid_path: str) -> Tuple[int, float, Tuple[int, int]]:
    cap = cv2.VideoCapture(vid_path)
    # Get the number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the frames per second (fps) of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Get the resolution of the video
    resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return num_frames, fps, resolution


def create_yolo_video(vid_path: str, csv_table_path: str, output_vid_path: str = None,
                      col_to_print: str = DEFAULT_COL_TO_PRINT) -> None:
    if output_vid_path is None:
        output_vid_path = os.path.join(os.path.dirname(vid_path),"output.mp4")
    num_frames, fps, resolution = get_video_info(vid_path)
    print(f"Input video: {vid_path}")
    print(f"Input video Resolution: {resolution}, fps: {fps}, num_frames: {num_frames}")
    print(f"Output video path {output_vid_path}")

    df = pd.read_csv(csv_table_path)
    # Create a video writer to save the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_vid_path, fourcc, fps, resolution)

    cap = cv2.VideoCapture(vid_path)
    for _ in tqdm(range(num_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Get the frame number
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Get the objects in the current frame
        objects_in_frame = df[df[FRAME_COL] == frame_number]

        # Draw a bounding box and label around each object
        for index, obj in objects_in_frame.iterrows():
            x, y, w, h = [int(coord) for coord in obj[[TOP_LEFT_X_COL, TOP_LEFT_Y_COL, WIDTH_COL, HEIGHT_COL]]]
            xmin, ymin = x, y
            xmax, ymax = x + w, y + h
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), BBOX_COLOR, BBOX_LINE_WIDTH)
            cv2.putText(frame, str(obj[col_to_print]), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, TEXT_COLOR, TEXT_LINE_WIDTH)

        # Write the frame to the output video
        out.write(frame)

    # Release the video writer and video capture
    out.release()
    cap.release()


if __name__ == '__main__':
    video_path = r"/home/zvi/Projects/DeepVehicleColorClassification/RawVideos/TEL AVIV Driving in ISRAEL 2021 4K • נסיעה בתל אביב.mp4"
    csv_path = "data.csv"
    create_yolo_video(video_path, csv_path, output_vid_path='./output.mp4')

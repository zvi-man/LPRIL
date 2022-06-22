from time import sleep
import cv2
from datetime import datetime
import matplotlib.pyplot as plt


# Constants
RTSP_CAM = r"rtsp://10.100.102.20:8554/mjpeg/1"
VIDEO = r"videos/Home Invasion Caught on 16 Surveillance Cameras.mp4"
WEB_CAM = 0
NUM_OF_FRAMES_TO_SKIP = 1
PLT_DELAY_TIME = 0.02


def main():
    image_count = 0
    vidcap = cv2.VideoCapture(WEB_CAM)
    success, image = vidcap.read()
    old_time = datetime.now()
    ax = plt.imshow(image)
    plt.ion()
    while success:
        success, image = vidcap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_count += NUM_OF_FRAMES_TO_SKIP
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, image_count)
        frame_delay = datetime.now() - old_time
        old_time = datetime.now()
        ax.set_data(image)
        plt.title(f"frame delay: {frame_delay}")
        plt.pause(PLT_DELAY_TIME)

    plt.ioff()


if __name__ == '__main__':
    main()

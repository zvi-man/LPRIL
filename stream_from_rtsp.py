from time import sleep
import cv2
from datetime import datetime
import matplotlib.pyplot as plt


def main():
    vidcap = cv2.VideoCapture(r"rtsp://10.100.102.20:8554/mjpeg/1")
    success, image = vidcap.read()
    old_time = datetime.now()
    ax = plt.imshow(image)
    plt.ion()
    while success:
        success, image = vidcap.read()
        frame_delay = datetime.now() - old_time
        old_time = datetime.now()
        ax.set_data(image)
        plt.title(f"frame delay: {frame_delay}")
        plt.pause(0.02)


if __name__ == '__main__':
    main()

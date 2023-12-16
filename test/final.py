import torchreid

import threading
from feature import match
from yolov8_ver2 import save_detections_from_video

video_path = "../dataset/test img/testme.mp4"
save_dir = "../dataset/test img/detections"
step = 60
roi = [204, 216, 1781, 1080]

if __name__ == "__main__":
    save_detections = threading.Thread(
        target=save_detections_from_video,
        args=(
            video_path,
            save_dir,
            step,
            roi,
        ),
    )

    save_detections.start()

    save_detections.join()
    match("../../dataset/ncu")

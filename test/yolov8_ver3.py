import numpy as np
import cv2
import torch

import os
import time
import supervision as sv
from feature import match_img
from collections import defaultdict
from ultralytics import YOLO


def adjust_brightness(image, alpha=1.1, beta=20):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def save_detections_from_video(
    video_path, save_dir, step=10, roi=[0, 0, 1920, 1080], min_detection_size=100
):
    model = YOLO("yolov8s.pt").to("cuda")  # 確保模型在 GPU 上運行
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    frame_index = 0
    last_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        if frame_index % step == 0:
            roi_frame = frame[roi[1] : roi[3], roi[0] : roi[2]]

            results = model(frame, imgsz=1280)[0]
            detections = sv.Detections.from_yolov8(results)
            detections = detections[detections.class_id == 0]  # Filter for person class
            detections = detections[detections.confidence > 0.7]

            frame_dir = os.path.join(save_dir, "gallery")
            os.makedirs(frame_dir, exist_ok=True)

            for item in os.listdir(frame_dir):
                os.remove(os.path.join(frame_dir, item))

            for j, det in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, det[:4])
                if (x2 - x1) * (y2 - y1) >= min_detection_size:
                    cropped_person = adjust_brightness(frame[y1:y2, x1:x2])
                    cv2.imwrite(
                        os.path.join(frame_dir, f"{x1}_{y1}_{x2}_{y2}.jpg"),
                        cv2.resize(cropped_person, (64, 128)),
                    )

            distmat, qf_list, gf_list = match_img("../dataset/test img")
            if not gf_list:
                frame_index += 1
                continue

            distmat_min = distmat.min(axis=1)
            min_indices = torch.argmin(distmat, axis=1)

            # 將 min_indices 轉換為 PyTorch 張量
            min_indices_tensor = torch.tensor(
                min_indices, device=distmat_min.values.device
            )

            matches = torch.where(
                distmat_min.values > 300,
                torch.tensor(-1, device=distmat_min.values.device),  # 確保 -1 張量在正確的設備上
                min_indices_tensor,  # 使用轉換後的 min_indices_tensor
            )

            seen = set()
            matches = [-1 if (m in seen or seen.add(m)) else m for m in matches]

            last_detections.clear()
            for i, match in enumerate(matches):
                if match != -1:
                    x1, y1, x2, y2 = map(
                        int, os.path.basename(gf_list[match]).split(".")[0].split("_")
                    )
                    last_detections.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2,
                            distmat_min.values[i],
                            os.path.basename(qf_list[i]),
                        )
                    )

        for det in last_detections:
            cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Dist: {det[4]:.2f}",
                (det[0], det[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                f"Query: {det[5]}",
                (det[0], det[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        end = time.time()
        print(end - start)

        frame_index += 1
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:  # Esc to quit
            break

    cap.release()


save_detections_from_video(
    "../dataset/test img/testme1.mp4",
    "../dataset/test img",
    step=7,
    roi=[204, 216, 1781, 1080],
)

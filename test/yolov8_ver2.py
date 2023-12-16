import numpy as np
import cv2
import torch

import os
import time
import supervision as sv
from feature import match_img
from ultralytics import YOLO


def adjust_brightness(image, alpha=1.1, beta=20):
    # 調整圖像的亮度
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def save_detections_from_video(
    video_path, save_dir, step=10, roi=[0, 0, 1920, 1080], min_detection_size=100
):
    # Load the YOLO model
    model = YOLO("yolov8s.pt")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    frame_index = 0

    roi_area = [0, 0, 1920, 1080]

    last_detections = []

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no frames are left

        # Process every ith frame
        if frame_index % step == 0:
            roi_frame = frame[roi[1] : roi[3], roi[0] : roi[2]]
            # Perform detection
            results = model(frame, imgsz=1280)[0]
            detections = sv.Detections.from_yolov8(results)
            detections = detections[detections.class_id == 0]  # Filter for person class
            detections = detections[
                detections.confidence > 0.7
            ]  # Filter for confidence greater than 10%

            # Create a directory for the current frame
            frame_dir = os.path.join(save_dir, "gallery")

            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)

            for item in os.listdir(frame_dir):
                item_path = os.path.join(frame_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)

            # Save the detected people
            for j, det in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, det[:4])
                cropped_person = frame[y1:y2, x1:x2]
                if (x2 - x1) * (y2 - y1) >= min_detection_size:
                    adjusted_person = adjust_brightness(cropped_person)

                    cropped_person = cv2.resize(adjusted_person, (64, 128))

                    file_name = f"{x1}_{y1}_{x2}_{y2}.jpg"
                    cv2.imwrite(
                        os.path.join(frame_dir, file_name),
                        cropped_person,
                    )

            distmat, qf_list, gf_list = match_img("../dataset/test img")
            print(distmat)

            if gf_list == []:
                frame_index += 1
                continue

            distmat_min = distmat.min(axis=1)
            min_indices = torch.argmin(distmat, axis=1)

            # 應用條件
            # matches = torch.where(distmat_min > 300, -1, min_indices)
            matches_t = torch.where(
                distmat_min.values > 300, torch.tensor(-1, device="cuda:0"), min_indices
            )

            # matches = matches_t.cpu().numpy()
            # print(matches)

            for i, match in enumerate(matches_t):
                if match != -1:
                    # 檢查這個最小值是否在之前已經出現過
                    if match in matches_t[:i]:
                        matches_t[i] = -1

            last_detections.clear()

            for i, match in enumerate(matches_t):
                if match != -1:  # 確保有有效的匹配
                    query_filename = os.path.basename(qf_list[i])
                    gallery_filename = os.path.basename(gf_list[match])
                    x1, y1, x2, y2 = map(int, gallery_filename.split(".")[0].split("_"))
                    last_detections.append(
                        (x1, y1, x2, y2, distmat_min.values[i], query_filename)
                    )  # 保存座標、距離值和 query 圖片名稱

        for x1, y1, x2, y2, dist, query_name in last_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            dist_text = f"Dist: {dist:.2f}"  # 格式化距離值
            name_text = f"Query: {query_name}"  # query 圖片名稱
            cv2.putText(
                frame,
                dist_text,
                (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                frame,
                name_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        frame_index += 1  # Increment frame index

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

        if key == 32:
            cv2.waitKey(0)
            continue

    # Release the video object
    cap.release()


roi_area = [204, 216, 1781, 1080]

# Call the function with the video path and save directory
save_detections_from_video(
    "../dataset/test img/testme1.mp4",
    "../dataset/test img",
    step=7,
    roi=roi_area,
)

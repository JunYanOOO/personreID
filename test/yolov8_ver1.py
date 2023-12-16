import cv2

import os
import supervision as sv
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
            frame_dir = os.path.join(save_dir, str(frame_index))
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)

            # Save the detected people
            for j, det in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, det[:4])
                cropped_person = frame[y1:y2, x1:x2]
                if (x2 - x1) * (y2 - y1) >= min_detection_size:
                    adjusted_person = adjust_brightness(cropped_person)

                    cropped_person = cv2.resize(adjusted_person, (64, 128))

                    file_name = f"{j+1:04d}_c{1}s{1}_{frame_index:06d}_00.jpg"
                    cv2.imwrite(
                        os.path.join(frame_dir, file_name),
                        cropped_person,
                    )

        frame_index += 1  # Increment frame index

    # Release the video object
    cap.release()


roi_area = [204, 216, 1781, 1080]

# Call the function with the video path and save directory
save_detections_from_video(
    "../dataset/test img/testme.mp4",
    "../dataset/test img/detections",
    step=60,
    roi=roi_area,
)

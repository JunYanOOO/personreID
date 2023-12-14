import cv2
import os
import supervision as sv
from ultralytics import YOLO

video_input_path = 'C:/Users/asuka/Desktop/oscar/NEWVideo/testme.mp4'
video_output_path = 'C:/Users/asuka/Desktop/oscar/people'


def adjust_brightness(image, alpha=1.1, beta=20):
    # 調整圖像的亮度
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def calculate_center(x1, y1, x2, y2):
    # 計算矩形框的中心點坐標
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y

def save_detections_from_video(video_path, save_dir, step=10, min_detection_size=100, roi=[0, 0, 1920, 1080]):
    global detected_people
    
    # 加載YOLO模型
    model = YOLO('yolov8s.pt')
    
    # 打開視頻文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 確保保存目錄存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    frame_index = 0
    while True:
        # 讀取幀
        ret, frame = cap.read()
        if not ret:
            break  # 如果沒有剩餘的幀，則跳出循環
        
        # 每第i帧進行處理
        if frame_index % step == 0:
            # 在每一幀上畫出 ROI 範圍
            #cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
            
            # 僅在 ROI 內進行檢測
            roi_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
            results = model(roi_frame, imgsz=1280)[0]
            detections = sv.Detections.from_yolov8(results)
            detections = detections[detections.class_id == 0]  # 過濾人物類別
            detections = detections[detections.confidence > 0.7]  # 過濾置信度大於90%
            
            # Create a directory for the current frame
            frame_dir = os.path.join(save_dir, str(frame_index))
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
                
            # 保存檢測到的人
            # Save the detected people
            for j, det in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, det[:4])
                cropped_person = roi_frame[y1:y2, x1:x2]
                if (x2 - x1) * (y2 - y1) >= min_detection_size:
                    # Adjust brightness
                    adjusted_person = adjust_brightness(cropped_person)
                    
                    # Resize the image
                    adjusted_person = cv2.resize(adjusted_person, (64, 128))
                    
                    # Save the adjusted image
                    cv2.imwrite(os.path.join(frame_dir, f"{j+1:04d}_c{1}s{1}_{frame_index}_00.jpg"), adjusted_person)
        
        frame_index += 1  # Increment frame index
        # 顯示處理後的幀
       # cv2.imshow("Video with ROI", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
        
        frame_index += 1  # 增加幀索引

    # 釋放視頻對象
    cap.release()
    cv2.destroyAllWindows()

# 調用函數並傳遞視頻路徑和保存目錄
if __name__ == "__main__":
    roi_area = [204, 216, 1781, 1080]  # 設定 ROI 的左上角和右下角坐標
    save_detections_from_video(video_input_path, video_output_path, roi=roi_area)
    

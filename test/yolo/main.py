import numpy as np
import supervision as sv
from ultralytics import YOLO
import uuid
from save_f import save_detections_from_video
from file import file_change
from change_file_name import rename_image
from create_floder import preprocess_images_randomly


############改檔案位置處############

#輸入影片
video_input_path = 'C:/Users/asuka/Desktop/oscar/NEWVideo/testme2.mp4'
#輸出影片
video_output_path = 'C:/Users/asuka/Desktop/oscar/NEWVideo/output02.mp4'
#輸出照片路徑
img_output_path = 'C:/Users/asuka/Desktop/oscar/people_o'
#照片源文件夹路径
source_folder = 'C:/Users/asuka/Desktop/oscar/people_o'
#照片目标文件夹路径
target_folder = 'C:/Users/asuka/Desktop/oscar/people_o_2'
# 改檔名
directory_path = 'C:/Users/asuka/Desktop/oscar/people_o_2'


############改檔案位置處############


person_id = []
labels = []
coordinates = []

class CountObject:
    def __init__(self, input_video_path, output_video_path):
        # 加载 YOLOv8 模型
        self.model = YOLO('yolov8s.pt')

        # 輸入視頻, 輸出視頻
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        # 視頻信息
        self.video_info = sv.VideoInfo.from_video_path(input_video_path)

        # 檢測框屬性(可刪除
        self.box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        
    def process_coordinates(self, detection, i):
        # 處理座標

        # 獲取檢測框的座標
        x, y, width, height = detection.xyxy
        
        coordinates.append({"person_id": person_id[i], "x": x, "y": y, "width": width, "height": height})

        print(f"Person {person_id[i]} - x: {x}, y: {y}, width: {width}, height: {height}")
        return x, y, width, height
    def process_frame(self, frame: np.ndarray, _) -> np.ndarray:
        # 檢測
        results = self.model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_yolov8(results)
    
        # 只保留人員
        detections = detections[detections.class_id == 0]
        detections = detections[detections.confidence > 0.7]

        # 為每個人分配ID
        global person_id
        global labels

        # 繪製檢測框
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        
        for i, detection in enumerate(detections):
            try:
                person_id.append(uuid.uuid4())
                xyxy, confidence, class_id, _,_ = detection
                label = f"Person {person_id[i]} {confidence:.2f}" if confidence is not None else f"Person {person_id[i]}"
                labels.append(label)
                # 處理座標
        
                # 獲取檢測框的座標
                x, y, width, height = xyxy
                
                coordinates.append({"person_id": person_id[i], "x": x, "y": y, "width": width, "height": height})
        
                print(f"Person {person_id[i]} - x: {x}, y: {y}, width: {width}, height: {height}")
   
            except ValueError:
                # 忽略解包失敗的 detection
                pass

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return frame


    def process_video(self):
        # 使用執行緒來執行 process_frame 方法
        sv.process_video(source_path=self.input_video_path, target_path=self.output_video_path,
                         callback=self.process_frame)
        # 在這裡你可以進一步處理所有座標資訊

        for coord in coordinates:
            print(f"Person {coord['person_id']} - x: {coord['x']}, y: {coord['y']}, width: {coord['width']}, height: {coord['height']}")



if __name__ == "__main__":
    obj = CountObject(video_input_path, video_output_path)
    obj.process_video()
    #roi_area = [204, 216, 1781, 1080]  # 設定 ROI 的左上角和右下角坐標
    roi_area = [0,0, 1920, 1080]  # 設定 ROI 的左上角和右下角坐標
    save_detections_from_video(video_input_path, img_output_path,step=10, min_detection_size=100, roi=roi_area)
    
    new_name_template = "0004_c2s4"  # 替換為你想要的新名稱前綴
    file_change(source_folder,target_folder )
    rename_image(directory_path)
    '''
    source_folder = "../dataset/ncu/all_people"
    target_folder = "../dataset/ncu"
    preprocess_images_randomly(source_folder, target_folder)
    '''
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 16:10:04 2023

@author: asuka
"""

import cv2

def capture_and_save_video(output_file='C:/Users/asuka/Desktop/oscar/NEWVideo/output_video.mp4', capture_device=0, width=640, height=480, fps=20):
    # 設定視頻擷取裝置
    cap = cv2.VideoCapture(capture_device)

    # 設定影片編碼器及其參數
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4編碼器
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while True:
        # 讀取一幀
        ret, frame = cap.read()

        # 如果成功讀取一幀，則顯示並寫入到影片文件
        if ret:
            cv2.imshow('Video', frame)
            out.write(frame)

        # 按下 'q' 鍵退出迴圈
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    capture_and_save_video()
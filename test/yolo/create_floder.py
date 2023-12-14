# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:06:55 2023

@author: asuka
"""

import os
import random
import shutil
from math import floor
from PIL import Image


def convert_to_jpg(image_path, target_folder):
    """
    將圖片轉換為JPG格式並保存到指定的目標文件夾中。
    """
    try:
        with Image.open(image_path) as img:
            # 移除擴展名並加上.jpg
            target_path = os.path.join(
                target_folder,
                os.path.splitext(os.path.basename(image_path))[0] + ".jpg",
            )
            img.convert("RGB").save(target_path, "JPEG")
    except Exception as e:
        print(f"無法轉換圖片 {image_path}: {e}")


def preprocess_images_randomly(source_folder, target_folder):
    """
    隨機預處理AI訓練用的圖像。從來源文件夾的子文件夾中將圖像複製到'gallery'文件夾中，
    並隨機將整個子文件夾分配到'train'和'query'文件夾中，比例大約為5:1。

    :param source_folder: 包含圖像的子文件夾的來源文件夾路徑
    :param target_folder: 將創建'gallery'、'train'和'query'文件夾的目標文件夾路徑
    """
    # 創建目標子目錄
    gallery_folder = os.path.join(target_folder, "gallery")
    train_folder = os.path.join(target_folder, "train")
    query_folder = os.path.join(target_folder, "query")

    for folder in [gallery_folder, train_folder, query_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 獲取所有子文件夾
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]
    random.shuffle(subfolders)

    # 分配子文件夾到train和query
    train_split = floor(len(subfolders) * (5 / 6))  # 5/6 分配給訓練
    train_subfolders = subfolders[:train_split]
    query_subfolders = subfolders[train_split:]

    # 處理每個子文件夾
    for subfolder in train_subfolders:
        images = [f.path for f in os.scandir(subfolder) if f.is_file()]
        for image in images:
            convert_to_jpg(image, train_folder)  # 轉換並複製到train文件夾
            convert_to_jpg(image, gallery_folder)  # 轉換並複製到gallery文件夾

    for subfolder in query_subfolders:
        images = [f.path for f in os.scandir(subfolder) if f.is_file()]
        for image in images:
            convert_to_jpg(image, query_folder)  # 轉換並複製到query文件夾
            convert_to_jpg(image, gallery_folder)  # 轉換並複製到gallery文件夾

    print("隨機預處理完成。")
    
if __name__ == "__main__":
    
    # 示範使用
    source_folder = "../dataset/ncu/all_people"
    target_folder = "../dataset/ncu"
    preprocess_images_randomly(source_folder, target_folder)

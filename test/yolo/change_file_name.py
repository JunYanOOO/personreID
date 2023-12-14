# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:07:42 2023

@author: asuka
"""

import os
import math
import random


def batch_rename_images(directory_path, new_name_template):
    # 獲取目錄下所有檔案的列表
    files = os.listdir(directory_path)
    # 根據文件名中的數字進行排序，這裡假定文件名格式如 '0042_c2s4_0_00.jpg'
    files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

    # 重命名文件
    for index, file in enumerate(files):
        print(file)
        # 分離文件擴展名
        extension = os.path.splitext(file)[1]
        # 創建新的文件名
        new_filename = f"{new_name_template}_{index:06d}_00{extension}"
        # 定義原始和新的文件完整路徑
        original_path = os.path.join(directory_path, file)
        new_path = os.path.join(directory_path, new_filename)
        # 重命名文件
        os.rename(original_path, new_path)
        print(f"Renamed '{file}' to '{new_filename}'")  # 打印出更改信息


def rename_image(directory_path):
    directory_path = os.path.join(os.getcwd(), directory_path)
    camera_ids = ["c1", "c2", "c3", "c4", "c5", "c6"]

    # 遍歷每個子資料夾
    for folder in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder)
        if os.path.isdir(folder_path):
            # 對該資料夾內的每個檔案進行操作
            for file in os.listdir(folder_path):
                original_path = os.path.join(folder_path, file)
                extension = os.path.splitext(file)[1]
                parts = file.split("_")

                folder = folder.zfill(4)
                parts[2] = parts[2].zfill(6)

                camera_id = random.choice(camera_ids)

                # 只修改檔名的第一部分，保持其他部分不變
                new_filename = f"{folder}_{camera_id}s1_{parts[2]}_{parts[3]}"
                new_path = os.path.join(folder_path, new_filename)
                os.rename(original_path, new_path)
                print(f"Renamed '{file}' to '{new_filename}'")  # 打印出更改信息


if __name__ == "__main__":
    # 設定包含子資料夾的根目錄路徑
    directory_path = "../dataset/people_r"
    new_name_template = "0004_c2s4"  # 替換為你想要的新名稱前綴

    rename_image(directory_path)
    # batch_rename_images(directory_path, new_name_template)
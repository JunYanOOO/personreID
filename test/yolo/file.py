# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:50:04 2023

@author: asuka
"""

import os
import shutil

# 源文件夹路径
source_folder = 'C:/Users/asuka/Desktop/oscar/people_o'

# 目标文件夹路径
target_folder = 'C:/Users/asuka/Desktop/oscar/people_o_2'
def file_change (source_folder, target_folder):
    # 遍历每个文件夹
    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        
        # 确保是文件夹
        if os.path.isdir(folder_path):
            # 检查文件夹是否包含照片
            photos_in_folder = [filename for filename in os.listdir(folder_path) if filename.endswith('.png') or filename.endswith('.jpg')]
            
            # 如果文件夹没有照片，跳过
            if not photos_in_folder:
                continue
    
            # 遍历文件夹内的照片
            for filename in photos_in_folder:
                # 提取照片编号，假设文件名格式是 "0001_c1s1_385_00.png"
                photo_number = filename.split('_')[0]
                
                # 目标文件夹的路径
                target_subfolder = os.path.join(target_folder, photo_number)
                
                # 如果目标文件夹不存在，创建它
                if not os.path.exists(target_subfolder):
                    os.makedirs(target_subfolder)
                
                # 将照片移动到目标文件夹
                shutil.move(os.path.join(folder_path, filename), os.path.join(target_subfolder, filename))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchreid import metrics
from torchreid.utils import FeatureExtractor

import os
import time


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            images.append(os.path.join(folder, filename))
    return images


def match_img(path):
    extractor = FeatureExtractor(
        model_name="osnet_x1_0",
        model_path="log/osnet__x1_0_market1501_softmax/model.pth.tar-25",
        device="cuda",
    )

    qf_list = load_images_from_folder(path + "/query")
    gf_list = load_images_from_folder(path + "/gallery")

    if gf_list == []:
        return [], [], []

    qf = extractor(qf_list)
    gf = extractor(gf_list)

    distmat = metrics.compute_distance_matrix(qf, gf)

    distmat_np = distmat.cpu().numpy()

    return distmat_np, qf_list, gf_list


# 顯示圖片函數
def show_matched_images(query_paths, gallery_paths, matches):
    for query_path, match in zip(query_paths, matches):
        query_img = Image.open(query_path)
        gallery_img = Image.open(gallery_paths[match])

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(query_img)
        plt.title("Query Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(gallery_img)
        plt.title("Matched Gallery Image")
        plt.axis("off")

        plt.show(block=False)
        plt.pause(0.1)  # 略微暫停，以確保圖片顯示

        print("Press 'space' to continue, or 'q' to quit.")
        if plt.waitforbuttonpress():
            plt.close()
            key = plt.get_current_fig_manager().canvas.key_press_handler_id
            if key == "q" or key == "Q":
                break


if __name__ == "__main__":
    match_img("../dataset/test img")

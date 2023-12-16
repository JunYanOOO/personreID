from __future__ import print_function, absolute_import
import numpy as np

import os.path as osp

__all__ = ["find_most_similar_images"]


def find_most_similar_images(distmat, dataset, topk=1, threshold=None):
    """
    返回每個查詢對象最相似的 topk 個畫廊圖像的名稱。

    Args:
        distmat (numpy.ndarray): 距離矩陣，形狀為 (num_query, num_gallery)。
        dataset (tuple): 包含 (query, gallery) 的元組，每個元素都是包含 (img_path(s), pid, camid, dsetid) 的元組。
        topk (int, optional): 要返回的最相似圖像數量。預設為 1。

    Returns:
        list of tuples: 每個查詢的最相似圖像名稱列表，每個元組包含 (query_img_name, [topk_similar_gallery_img_names])。
    """
    num_q, num_g = distmat.shape
    query, gallery = dataset

    assert num_q == len(query)
    assert num_g == len(gallery)

    # indices = np.argsort(distmat, axis=1)

    # similar_images = []
    # for q_idx in range(num_q):
    #     qimg_path = query[q_idx][0]
    #     qimg_name = osp.basename(qimg_path)

    #     topk_indices = indices[q_idx, :topk]
    #     topk_gallery_paths = []
    #     for g_idx in topk_indices:
    #         if threshold is None or distmat[q_idx, g_idx] <= threshold:
    #             topk_gallery_paths.append(gallery[g_idx][0])
    #             topk_gallery_names = [osp.basename(path) for path in topk_gallery_paths]
    #             similar_images.append((qimg_name, topk_gallery_names))
    #         else:
    #             similar_images.append((qimg_name, [0]))

    # return similar_images

    ##########################################3
    # 運用 NumPy 的向量化操作來提高效率
    if threshold is not None:
        distmat = np.where(distmat <= threshold, distmat, np.inf)

    indices = np.argsort(distmat, axis=1)[:, :topk]

    similar_images = []
    for q_idx in range(num_q):
        qimg_name = osp.basename(query[q_idx][0])
        topk_gallery_names = [
            osp.basename(gallery[g_idx][0]) for g_idx in indices[q_idx]
        ]
        similar_images.append((qimg_name, topk_gallery_names))

    return similar_images

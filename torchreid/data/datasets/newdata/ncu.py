from __future__ import absolute_import, print_function, division
import re
import glob
import os.path as osp

from ..dataset import ImageDataset

class NCU(ImageDataset):
    dataset_dir = 'ncu'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # 假设有三个子目录分别对应于train, query, gallery
        train_dir = osp.join(self.dataset_dir, 'train')
        query_dir = osp.join(self.dataset_dir, 'query')
        gallery_dir = osp.join(self.dataset_dir, 'gallery')

        # 使用glob从每个子目录读取图像路径
        train = self.process_dir(train_dir, relabel=True)
        query = self.process_dir(query_dir, relabel=False)
        gallery = self.process_dir(gallery_dir, relabel=False)

        super(NCU, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        pattern = re.compile(r'(\d{4})_c(\d)s(\d)_\d{6}_\d{2}.jpg')
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pid_container = set()
        for img_path in img_paths:
            pid, camid, _ = map(int, pattern.search(osp.basename(img_path)).groups())
            if relabel:
                pid_container.add(pid)
        pid2label = {pid: idx for idx, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid, _ = map(int, pattern.search(osp.basename(img_path)).groups())
            if relabel:
                pid = pid2label[pid]
            camid -= 1  # assume the camid starts from 1
            data.append((img_path, pid, camid))
        return data


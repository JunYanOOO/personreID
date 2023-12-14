import torchreid

# 初始化数据管理器
datamanager = torchreid.data.ImageDataManager(
    root='../dataset', # 指定数据集的根目录
    sources='ncu', # 指定数据集名称
    height=256, 
    width=128, 
    batch_size_train=32, 
    batch_size_test=100, 
    transforms=['random_flip', 'random_crop']
)



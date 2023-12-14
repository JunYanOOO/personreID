import torchreid

import time


def main():
    datamanager = torchreid.data.ImageDataManager(
        root="../dataset",
        sources="market1501",
        targets="ncu",
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
    )

    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True,
    )

    model_path = "log\osnet_x1_0_market1501_softmax_cosinelr\model\model.pth.tar-250"
    torchreid.utils.load_pretrained_weights(model, model_path)

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(model, optim="adam", lr=0.0003)

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler="single_step", stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager, model, optimizer=optimizer, scheduler=scheduler, label_smooth=True
    )

    # engine.run(
    #     save_dir="log/resnet50",
    #     max_epoch=0,
    #     eval_freq=-1,
    #     print_freq=10,
    #     test_only=True,
    #     name_output=True,
    # )

    mathch_name = engine.test2(
        dist_metric="euclidean",
        normalize_feature=False,
        visrank=True,
        visrank_topk=1,
        save_dir="log/resnet50",
        use_metric_cuhk03=False,
        ranks=[1],
        rerank=True,
        name_output=True,
        threshold=100,
    )

    for name in mathch_name:
        print(name)


if __name__ == "__main__":
    main()

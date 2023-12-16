import torch

import torchreid

import time
import cProfile


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
        name="osnet_x0_25",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True,
    )

    model_path = "log\osnet_x0_25_market1501_softmax\model\model.pth.tar-25"
    torchreid.utils.load_pretrained_weights(model, model_path)

    model = model.cuda()
    # 跑一次熱機
    # model(torch.empty(1, 3, 64, 128, dtype=torch.float64, device=torch.device("cuda")))

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
        visrank=False,
        visrank_topk=1,
        save_dir="log/resnet50",
        use_metric_cuhk03=False,
        ranks=[1],
        rerank=False,
        name_output=True,
        threshold=100,
    )

    for name in mathch_name:
        print(name)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print(f"total:{end - start}")

    # cProfile.run("main()")

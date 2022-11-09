import torch

torch.cuda.current_device()
torch.cuda._initialized = True
import argparse
import os
from datetime import datetime

import torch.nn.functional as F
from torch.autograd import Variable
from lib.C2FNet import (
    BasicACFMC2FNet,
    BasicACFMDGCMC2FNet,
    BasicC2FNet,
    BasicCIMC2FNet,
    BasicDGCMC2FNet,
    C2FNet,
    C2FNetWOMSCA,
)
from utils.AdaX import AdaXW
from utils.dataloader import get_loader
from utils.utils import AvgMeter, adjust_lr, clip_gradient
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from tqdm import tqdm

# from ..BBSC2F_P1.lib.BBS_C2F import BBS_C2FNet


model_registry = {
    "C2FNet": C2FNet,
    "BasicC2FNet": BasicC2FNet,
    "BasicCIMC2FNet": BasicCIMC2FNet,
    "BasicACFMC2FNet": BasicACFMC2FNet,
    "BasicDGCMC2FNet": BasicDGCMC2FNet,
    "BasicACFMDGCMC2FNet": BasicACFMDGCMC2FNet,
    "C2FNetWOMSCA": C2FNetWOMSCA,
}


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def LCE_loss(pred1, pred2, mask):
    loss1 = structure_loss(pred1, mask)
    loss2 = structure_loss(pred2, mask)
    loss = loss1 + loss2
    return loss


def train_cifar(
    config, model, epochs, save_path, checkpoint_dir=None
):
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    model.train()
    params = model.parameters()
    optimizer = AdaXW(
        params,
        config["lr"],
        weight_decay=config["weight_decay"],
    )
    file_name = "tuning_results.txt"
    with open(file_name, "a") as log_file:
        log_file.write("\n\n" + "new set of params \n" + str(config) + "\n")

    for epoch in range(1, epochs):
        adjust_lr(optimizer, config["lr"], epoch, config["decay_rate"], config["decay_epoch"])
        # ---- multi-scale training ----
        size_rates = [0.75, 1, 1.25]
        loss_record3 = AvgMeter()
        with tqdm(range(1, len(train_loader)+1)) as pbar:
            for i, pack in enumerate(train_loader, start=1):
                for rate in size_rates:
                    optimizer.zero_grad()
                    # ---- data prepare ----
                    images, gts = pack
                    images = Variable(images).cuda()
                    gts = Variable(gts).cuda()
                    # ---- rescale ----
                    trainsize = int(round(opt.trainsize * rate / 32) * 32)
                    if rate != 1:
                        images = F.upsample(
                            images,
                            size=(trainsize, trainsize),
                            mode="bilinear",
                            align_corners=True,
                        )
                        gts = F.upsample(
                            gts,
                            size=(trainsize, trainsize),
                            mode="bilinear",
                            align_corners=True,
                        )
                    # ---- forward ----
                    pred1, pred2 = model(images)
                    # ---- loss function ----
                    loss3 = LCE_loss(pred1, pred2, gts)
                    loss = loss3
                    # ---- backward ----
                    loss.backward()
                    clip_gradient(optimizer, opt.clip)
                    optimizer.step()
                    # ---- recording loss ----
                    if rate == 1:
                        loss_record3.update(loss3.data, opt.batchsize)
                # ---- train visualization ----

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Last_loss": f"{loss_record3.show():.2f}",
                    }
                )    
                if i % 20 == 0 or i == total_step:
                    test_result = "{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], [lateral-3: {:.4f}]".format(
                        datetime.now(), epoch, opt.epoch, i, total_step, loss_record3.show()
                    )
                    with open(file_name, "a") as log_file:
                        log_file.write(test_result + "\n")
        visual = {"time": datetime.now(), "Epoch ": epoch, "loss": loss_record3.show()}
        os.makedirs(save_path, exist_ok=True)

        tune.report(loss = visual["loss"])
        if (epoch + 1) % 5 == 0:
            model_name = f"C2FNet-{config['lr']}-{config['weight_decay']}-{config['decay_rate']}--{config['decay_epoch']}-{epoch}.pth"
            torch.save(model.state_dict(), save_path + model_name)
            print("[Saving Snapshot:]", save_path + model_name)
            # with tune.checkpoint_dir(epoch) as checkpoint_dir:
            #     path = os.path.join(checkpoint_dir, "check")
            #     torch.save((model.state_dict(), optimizer.state_dict()), path)

    print("Current hyperparameters set training finished \n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=200, help="epoch number")
    parser.add_argument("--batchsize", type=int, default=4, help="training batch size")
    parser.add_argument(
        "--trainsize", type=int, default=352, help="training dataset size"
    )
    parser.add_argument(
        "--clip", type=float, default=0.5, help="gradient clipping margin"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/TrainDataset",
        help="path_to_train_dataset",
    )
    parser.add_argument("--train_save", type=str, default="C2FNet")
    parser.add_argument("--model", default="C2FNet")
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = model_registry[opt.model]()

    image_root = "{}/Imgs/".format(opt.train_path)
    gt_root = "{}/GT/".format(opt.train_path)
    print(f"image root: {image_root}")
    print(f"gt root: {gt_root}")

    train_loader = get_loader(
        image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize
    )
    total_step = len(train_loader)

    print("Start Tuning \n")

    config = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-4, 1e-1),
        "decay_rate": tune.choice([0.05, 0.1, 0.2]),
        "decay_epoch": tune.choice([30, 40, 50])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["lr", "weight_decay", "decay_rate", "decay_epoch"],
        metric_columns=["loss", "training_iteration"])
    
    gpus_per_trial = 1

    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model.cuda()

    result = tune.run(
        tune.with_parameters(train_cifar, model=model, epochs=opt.epoch, save_path=opt.train_save),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=20,
        scheduler=scheduler,
        progress_reporter=reporter,
        max_concurrent_trials=2
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final loss: {}".format(
        best_trial.last_result["loss"]))




import numpy as np
import torch

torch.cuda.current_device()
torch.cuda._initialized = True
import argparse
import os
from datetime import datetime

import torch.nn.functional as F
from PIL import Image
from ray import tune
from ray.tune import CLIReporter, ExperimentAnalysis, register_trainable
from ray.tune.schedulers import ASHAScheduler
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

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
from utils.dataloader import get_loader, test_dataset
from utils.utils import AvgMeter, adjust_lr, clip_gradient

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


def Eval_Smeasure(preds, gts):
    def _S_object(pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = _object(fg, gt)
        o_bg = _object(bg, 1 - gt)
        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        return score

    def _S_region(pred, gt):
        X, Y = _centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
        p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
        Q1 = _ssim(p1, gt1)
        Q2 = _ssim(p2, gt2)
        Q3 = _ssim(p3, gt3)
        Q4 = _ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        return Q

    def _centroid(gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            X = torch.eye(1).cuda() * round(cols / 2)
            Y = torch.eye(1).cuda() * round(rows / 2)
        else:
            total = gt.sum()
            i = torch.from_numpy(np.arange(0, cols)).cuda().float()
            j = torch.from_numpy(np.arange(0, rows)).cuda().float()
            X = torch.round((gt.sum(dim=0) * i).sum() / total + 1e-20)
            Y = torch.round((gt.sum(dim=1) * j).sum() / total + 1e-20)
        return X.long(), Y.long()

    def _divideGT(gt, X, Y):
        h, w = gt.size()[-2:]
        area = h * w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

    alpha, avg_q, img_num = 0.5, 0.0, 0.0
    for pred, gt in zip(preds, gts):
        pred = (pred - torch.min(pred)) / (torch.max(pred) - torch.min(pred) + 1e-20)
        y = gt.mean()
        if y == 0:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            Q = x
        else:
            gt[gt >= 0.5] = 1
            gt[gt < 0.5] = 0
            Q = alpha * _S_object(pred, gt) + (1 - alpha) * _S_region(pred, gt)
            if Q.item() < 0:
                Q = torch.FloatTensor([0.0])
        img_num += 1.0
        avg_q += Q.item()
    avg_q /= img_num
    return avg_q


def Eval_mae(preds, gts):
    avg_mae, img_num = 0.0, 0.0
    with torch.no_grad():
        trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in zip(preds, gts):
            mea = torch.abs(pred - gt).mean()
            if mea == mea:  # for Nan
                avg_mae += mea
                img_num += 1.0
        avg_mae /= img_num
        return avg_mae.item()


def train_cifar(config, model, epochs, save_path, checkpoint_dir=None):
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

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
        model.train()
        adjust_lr(
            optimizer, config["lr"], epoch, config["decay_rate"], config["decay_epoch"]
        )
        # ---- multi-scale training ----
        size_rates = [0.75, 1, 1.25]
        loss_record3 = AvgMeter()
        with tqdm(range(1, len(train_loader) + 1)) as pbar:
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
                        datetime.now(),
                        epoch,
                        opt.epoch,
                        i,
                        total_step,
                        loss_record3.show(),
                    )
                    with open(file_name, "a") as log_file:
                        log_file.write(test_result + "\n")

        # validation
        model.eval()
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])

        os.makedirs(f"{save_path}/val_predictions", exist_ok=True)
        preds = []
        gts = []
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data(i)
            gt = np.asarray(gt, np.float32)
            image = image.cuda()
            torch.cuda.synchronize()
            _, res = model(image)
            torch.cuda.synchronize()

            res = F.upsample(res, size=gt.shape, mode="bilinear", align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # print(np.shape(res))
            _, fullflname = os.path.split(name)
            print("save img to: ", f"{save_path}/val_predictions/{fullflname}")
            res = res * 255
            res = Image.fromarray(res.astype(np.uint8))
            res.save(
                f"{save_path}/val_predictions/{fullflname}"
            )  # for backing up and checking

            pred = res.convert("L")
            pred = trans(res).cuda()
            gt = trans(gt).cuda()
            preds.append(pred)
            gts.append(gt)

        similairty = Eval_Smeasure(preds, gts)
        mea = Eval_mae(preds, gts)

        visual = {
            "time": datetime.now(),
            "Epoch ": epoch,
            "training_loss": loss_record3.show().cpu().item(),
            "val_mea": mea,
            "val_smeasure": similairty,
        }
        os.makedirs(save_path, exist_ok=True)

        tune.report(
            training_loss=visual["training_loss"],
            val_mea=visual["val_mea"],
            val_smeasure=visual["val_smeasure"],
        )
        if (epoch + 1) % 5 == 0:
            model_name = f"/C2FNet-{config['lr']}-{config['weight_decay']}-{config['decay_rate']}--{config['decay_epoch']}-{epoch}.pth"
            torch.save(model.state_dict(), save_path + model_name)
            print("[Saving Snapshot:]", save_path + model_name)

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
    parser.add_argument(
        "--validation_path",
        type=str,
        default="data/TestDataset",
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

    val_image_root = "{}/Imgs/".format(opt.validation_path)
    val_gt_root = "{}/GT/".format(opt.validation_path)
    print(f"val image root: {val_image_root}")
    print(f"val gt root: {val_gt_root}")

    train_loader = get_loader(
        image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize
    )
    test_loader = test_dataset(
        val_image_root, val_gt_root, opt.trainsize, sample_size=100
    )

    total_step = len(train_loader)

    print("Save validation gt images")
    os.makedirs("/export/home2/qishuai/ray_results/selected_gt", exist_ok=True)
    
    for i in tqdm(range(test_loader.size)):
        _, gt, name = test_loader.load_data(i)
        _, fullflname = os.path.split(name)
        print("save gt img to: ", f"/export/home2/qishuai/ray_results/selected_gt/{fullflname}")
        gt.save(f"/export/home2/qishuai/ray_results/selected_gt/{fullflname}")


    print("Start Tuning \n")

    config = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.choice([5e-3, 1e-2, 5e-2, 1e-1]),
        "decay_rate": tune.choice([0.05, 0.1, 0.2]),
        "decay_epoch": tune.choice([30, 40, 50]),
    }

    scheduler = ASHAScheduler(
        metric="training_loss", mode="min", max_t=20, grace_period=3, reduction_factor=2
    )
    reporter = CLIReporter(
        parameter_columns=["lr", "weight_decay", "decay_rate", "decay_epoch"],
        metric_columns=[
            "training_loss",
            "val_mea",
            "val_smeasure",
            "training_iteration",
        ],
    )

    gpus_per_trial = 1

    result = tune.run(
        tune.with_parameters(
            train_cifar, model=model, epochs=opt.epoch, save_path=opt.train_save
        ),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=20,
        scheduler=scheduler,
        progress_reporter=reporter,
        max_concurrent_trials=3,
    )

    print("printing result")

    # # re-analysis
    # register_trainable(
    #     "train_cifar",
    #     tune.with_parameters(
    #         train_cifar, model=model, epochs=opt.epoch, save_path=opt.train_save
    #     ),
    # )
    # analysis = ExperimentAnalysis(
    #     "/export/home2/qishuai/ray_results/train_cifar_2022-11-12_23-54-42"
    # )

    best_trial = result.get_best_trial("training_loss", mode="min")
    print(
        "Best trial config: {}".format(
            result.get_best_config("training_loss", mode="min")
        )
    )
    print("Best trial final loss: {}".format(best_trial.last_result))
    print(f"Best trail runner ip: {best_trial.get_runner_ip}")

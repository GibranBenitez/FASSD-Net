import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import pdb

from torch.utils import data
import sys

from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

def train(cfg, writer, logger, args):
    # cfg

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["train_split"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
    )

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        img_size=(1024,2048),
    )

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader, batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"]["n_workers"]
    )

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = FASSDNet(n_classes=19, alpha=args.alpha).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print( 'Parameters:',total_params )
    model.apply(weights_init)

    # Non-strict ImageNet pre-train
    pretrained_path='weights/imagenet_weights.pth'
    checkpoint = torch.load(pretrained_path)
    q = 1
    model_dict = {}
    state_dict = model.state_dict()

    # print('================== Weights orig: ', model.base[1].conv.weight[0][0][0])
    for k, v in checkpoint.items():  
        if q == 1:
            # print("===> Key of checkpoint: ", k)
            # print("===> Value of checkpoint: ", v[0][0][0])
            if ('base.'+k in state_dict):
                # print("============> CONTAINS KEY...")
                # print("===> Value of the key: ", state_dict['base.'+k][0][0][0])
                pass

            else:
                # print("============> DOES NOT CONTAIN KEY...")
                pass
            q = 0

        if ('base.'+k in state_dict) and (state_dict['base.'+k].shape == checkpoint[k].shape):
            model_dict['base.'+k] = v

    state_dict.update(model_dict)  # Updated weights with ImageNet pretraining
    model.load_state_dict(state_dict)
    # print('================== Weights loaded: ', model.base[0].conv.weight[0][0][0])

    # Multi-gpu model
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))   

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    print("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg)
    print("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            
            print_str = "Finetuning model from '{}'".format(cfg["training"]["finetune"])
            if logger is not None:
                logger.info(print_str)
            print(print_str)
            
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            
            print_str = "Loaded checkpoint '{}' (iter {})".format(cfg["training"]["resume"], checkpoint["epoch"])
            print(print_str)
            if logger is not None:
                logger.info(print_str)
        else:
            print_str = "No checkpoint found at '{}'".format(cfg["training"]["resume"])
            print(print_str)
            if logger is not None:
                logger.info(print_str)
    
    if cfg["training"]["finetune"] is not None:
        if os.path.isfile(cfg["training"]["finetune"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["finetune"])
            )
            checkpoint = torch.load(cfg["training"]["finetune"])
            model.load_state_dict(checkpoint["model_state"])

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True
    loss_all = 0
    loss_n = 0
    sys.stdout.flush()
    
    while i <= cfg["training"]["train_iters"] and flag:
        for (images, labels, _) in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)
            loss.backward()
            optimizer.step()
            c_lr = scheduler.get_lr()

            time_meter.update(time.time() - start_ts)
            loss_all += loss.item()
            loss_n += 1
            
            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}  lr={:.6f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss_all / loss_n,
                    time_meter.avg / cfg["training"]["batch_size"],
                    c_lr[0],
                )
                

                print(print_str)
                if logger is not None:
                    logger.info(print_str)
                    
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()

            if (i + 1) % cfg["training"]["val_interval"] == 0 or (i + 1) == cfg["training"][
                "train_iters"
            ]:
                torch.cuda.empty_cache()
                model.eval()
                loss_all = 0
                loss_n = 0
                with torch.no_grad():
                    # for i_val, (images_val, labels_val, _) in tqdm(enumerate(valloader)):
                    for i_val, (images_val, labels_val, _) in enumerate(valloader):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)

                        outputs = model(images_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)

                        pred = outputs.data.max(1)[1].cpu().numpy()
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss.item())

                writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)

                print_str = "Iter %d Val Loss: %.4f" % (i + 1, val_loss_meter.avg)
                if logger is not None:
                    logger.info(print_str)
                print(print_str)

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print_str = "{}: {}".format(k, v)
                    if logger is not None:
                        logger.info(print_str)
                    print(print_str)
                    
                    writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

                for k, v in class_iou.items():
                    print_str = "{}: {}".format(k, v)
                    if logger is not None:
                        logger.info(print_str)
                    print(print_str)
                    
                    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

                val_loss_meter.reset()
                running_metrics_val.reset()
                
                state = {
                      "epoch": i + 1,
                      "model_state": model.state_dict(),
                      "optimizer_state": optimizer.state_dict(),
                      "scheduler_state": scheduler.state_dict(),
                }
                save_path = os.path.join(
                    writer.file_writer.get_logdir(),
                    "{}_{}_checkpoint.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                )
                torch.save(state, save_path)

                if score["Mean IoU : \t"] >= best_iou:  # Save best model (mIoU)
                    best_iou = score["Mean IoU : \t"]
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(
                        writer.file_writer.get_logdir(),
                        "{}_{}_best_model.pkl".format(cfg["model"]["arch"], cfg["data"]["dataset"]),
                    )
                    torch.save(state, save_path)
                torch.cuda.empty_cache()

            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break
            sys.stdout.flush()  # Added

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    
    
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/FASSD-Net.yml",  # configs/FASSD-Net-L1.yml, configs/FASSD-Net-L2.yml
        help="Configuration file to use",
    )
        
    parser.add_argument(
        "--model",
        nargs="?",
        type=str,
        default="FASSDNet", # FASSDNetL1, FASSDNetL2
        help="Model variation to use",
    )
    
    parser.add_argument("--alpha", default=2, nargs="?", type=int)  # NEW ALPHA IMPLEMENTATION

    args = parser.parse_args()

    if args.model == "FASSDNet":
        from ptsemseg.models.FASSDNet import FASSDNet
    
    elif args.model == "FASSDNetL1":
        from ptsemseg.models.FASSDNetL1 import FASSDNet
        
    elif args.model == "FASSDNetL2":
        from ptsemseg.models.FASSDNetL2 import FASSDNet
        
    else:
        print("No valid model found")
        
    
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], "cur")
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)
    
    logger = None
    # logger = get_logger(logdir)
    # logger.info("Let the games begin")

    train(cfg, writer, logger, args)

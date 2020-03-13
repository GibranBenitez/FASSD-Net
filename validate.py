import yaml
import torch
import argparse
import timeit
import time
import os
import numpy as np
import scipy.misc as misc

from torch.utils import data

from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict

# from ptsemseg.models.FASSDNet import FASSDNet
# from ptsemseg.models.FASSDNetL1 import FASSDNet
# from ptsemseg.models.FASSDNetL2 import FASSDNet

torch.backends.cudnn.benchmark = True  ### Duplicated

def validate(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
        # split = "test", # if test set is used
        is_transform=True,
        img_size=(1024,2048),  
    )

    n_images = len(loader.files[cfg["data"]["val_split"]])
    # print("N images", len(loader.files[cfg["data"]["val_split"]]))  # or "test" instead of cfg["data"]["val_split"]

    n_classes = loader.n_classes
    valloader = data.DataLoader(loader, batch_size=1, num_workers=1)
    running_metrics = runningScore(n_classes)

    # Setup Model
    model = FASSDNet(19).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    
    model.eval()
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters: ', total_params )
    
    torch.backends.cudnn.benchmark=True

    for i, (images, labels, fname) in enumerate(valloader):
        start_time = timeit.default_timer()

        images = images.to(device)
        
        if i == 0:
          with torch.no_grad():
            outputs = model(images)        
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
          outputs = model(images)

        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start_time
            
        if args.save_image:
            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
            save_rgb = True
                
            decoded = loader.decode_segmap_id(pred)
            dir = "./out_predID/"
            if not os.path.exists(dir):
              os.mkdir(dir)
            misc.imsave(dir+fname[0], decoded)

            if save_rgb:
                decoded = loader.decode_segmap(pred)
                img_input = np.squeeze(images.cpu().numpy(),axis=0)
                img_input = img_input.transpose(1, 2, 0)
                blend = img_input * 0.2 + decoded * 0.8
                fname_new = fname[0]
                fname_new = fname_new[:-4]
                fname_new1 = fname_new + '.jpg'
                fname_new2 = fname_new + '.png'  # For Color labels

                dir = "./out_rgb/"
                if not os.path.exists(dir):
                  os.mkdir(dir)
                misc.imsave(dir+fname_new1, blend)

                # Save labels with color
                dir2 = "./out_color/"
                if not os.path.exists(dir2):
                  os.mkdir(dir2)
                misc.imsave(dir2+fname_new2, decoded)

                
        pred = outputs.data.max(1)[1].cpu().numpy()

        gt = labels.numpy()
        s = np.sum(gt==pred) / (1024*2048)
        
        running_metrics.update(gt, pred)
        print("iteration {}/{}".format(i, n_images), end='\r')

    score, class_iou = running_metrics.get_scores()
    # print("score", score)

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/FASSD-Net.yml",
        help="Config file to be used",
    )

    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="FASSD-Net-Cytiscapes.pkl",
        help="Path to the saved model",
    )    

    parser.add_argument(
        "--save_image",
        dest="save_image",
        action="store_true",
        help="Enable saving inference result image into out_img/ |\
                              False by default",
    )
    parser.set_defaults(save_image=False)
    
    parser.add_argument(
        "--model",
        nargs="?",
        type=str,
        default="FASSDNet", # FASSDNetL1, FASSDNetL2
        help="Model variation to use",
    )
    
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

    validate(cfg, args)

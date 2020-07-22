import torch
import argparse
import time
import os
import sys
import numpy as np
import pdb
import torch.nn.functional as F
from ptsemseg.utils import convert_state_dict
from PIL import Image

colors = [  # [  0,   0,   0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]

label_colours = dict(zip(range(19), colors))

valid_classes = [
    7, # "road"
    8, # "sidewalk"
    11, # "building"
    12, # "wall"
    13, # "fence"
    17, # "pole"
    19, # "traffic_light"
    20, # "traffic_sign"
    21, # "vegetation"
    22, # "terrain"
    23, # "sky"
    24, # "person"
    25, # "rider"
    26, # "car"
    27, # "truck"
    28, # "bus"
    31, # "train"
    32, # "motorcycle"
    33, # "bicycle"
]

def decode_segmap(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    
    return rgb

def decode_segmap_id(temp):
    ids = np.zeros((temp.shape[0], temp.shape[1]),dtype=np.uint8)
    for l in range(0, 19):  # n_classes
        ids[temp == l] = valid_classes[l]
    return ids

def transform(img, img_size=(1024,2048)):
    """transform
    :param img:
    """
    img = np.array(img.resize(
            (img_size[1], img_size[0])))  # uint8 with RGB mode
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)

    value_scale = 255
    mean = [0.406, 0.456, 0.485]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * value_scale for item in std]

    # if self.img_norm:
    img = (img - mean) / std

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)  # Needed to make a single prediction...
    img = torch.from_numpy(img).float()

    return img

def inference(args, model, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Model
    model = model.to(device)

    if args.model == "FASSDNet_trt" or args.model == "FASSDNetL1_trt" or args.model == "FASSDNetL2_trt":
        pass
    else:
        state = convert_state_dict(torch.load(model_path)["model_state"])
        model.load_state_dict(state)    
    
    model.eval()
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters: ', total_params )    
    torch.backends.cudnn.benchmark=True

    img_list = []  # list of paths
    if os.path.isdir(args.path):  
        d = args.path
        for path in os.listdir(d):
            full_path = os.path.join(d, path)
            if os.path.isfile(full_path):
                img_list.append(full_path)
    elif os.path.isfile(args.path):  
        img_list.append(args.path)
    else:  
        print("Not supported path type" )
        raise SystemExit(0)

    if args.eval_res != "original":
        h, w = map(int, args.eval_res.split(','))

    print("\n{} images to be processed at {}x{}:".format(len(img_list),h,w))
    
    for i, img_path in enumerate(img_list):
        imgPill = Image.open(img_path)
        img_input = np.array(imgPill) / 255.0
        orig_res = img_input.shape      

        if args.eval_res == "original":
            h = orig_res[0]
            w = orig_res[1]
        
        img = transform(imgPill, img_size=(h,w))    
        img = img.to(device)
        
        t_start = time.time()
        probs = model(img) 
        elapsed_time = time.time() - t_start
        print("  {} evaluated in {:.2f} ms".format(os.path.basename(img_path),elapsed_time*1000))

        probs = F.interpolate(probs,size=(orig_res[0], orig_res[1]), mode="bilinear", align_corners=True)
        pred = np.squeeze(probs.data.max(1)[1].cpu().numpy(), axis=0)
        decoded = decode_segmap(pred)

        res_dir = "./demo_results_{}/".format(args.model)
        if not os.path.exists(res_dir):
          os.mkdir(res_dir)
                
        blend = img_input * 0.4 + decoded * 0.6
        fname_new = img_path.split(os.sep)[-1][:-4]

        bimg = Image.fromarray(np.uint8(blend*255))
        bimg.save(res_dir+fname_new+"_blend_{}x{}.jpg".format(h,w))

        bdec = Image.fromarray(np.uint8(decoded*255))
        bdec.save(res_dir+fname_new+"_mask_{}x{}.png".format(h,w))
                
    print("Results saved at {}\n".format(res_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")        
    parser.add_argument("--model", type=str, default="FASSDNetL2_trt", help="Model used for the demo FASSDNetL2 (fastest)")
    parser.add_argument("--path", type=str, default="./demo_imgs/", help="path or image to evaluate")
    parser.add_argument("--eval_res", type=str, default="512,1024", help="input size of the network. 'original' evaluates with the original image size")
            
    args = parser.parse_args()
    print("")
    
    ## FASSDNet Models
    if args.model == "FASSDNet":
        from ptsemseg.models.FASSDNet import FASSDNet
        print("Model: FASSDNet")
        model = FASSDNet(19)
        model_path = './weights/FASSD-Net-Cityscapes.pkl'
    elif args.model == "FASSDNetL1":
        from ptsemseg.models.FASSDNetL1 import FASSDNet
        print("Model: FASSDNetL1")
        model = FASSDNet(19)
        model_path = './weights/FASSD-Net-L1-Cityscapes.pkl'
    elif args.model == "FASSDNetL2":
        from ptsemseg.models.FASSDNetL2 import FASSDNet
        print("Model: FASSDNetL2")
        model = FASSDNet(19)
        model_path = './weights/FASSD-Net-L2-Cityscapes.pkl'
    elif args.model == "FASSDNet_trt" or args.model == "FASSDNetL1_trt" or args.model == "FASSDNetL2_trt":
        if args.eval_res != "original":
            if args.model == "FASSDNet_trt":
                from ptsemseg.models.FASSDNet import FASSDNet
                model_path_non_trt = "./weights/FASSD-Net-Cityscapes.pkl"
                print("Model: FASSDNet w/trt")
                                
            elif args.model == "FASSDNetL1_trt":
                from ptsemseg.models.FASSDNetL1 import FASSDNet
                model_path_non_trt = "./weights/FASSD-Net-L1-Cityscapes.pkl"
                print("Model: FASSDNetL1 w/trt")
                
            elif args.model == "FASSDNetL2_trt":
                from ptsemseg.models.FASSDNetL2 import FASSDNet
                model_path_non_trt = "./weights/FASSD-Net-L2-Cityscapes.pkl"
                print("Model: FASSDNetL2 w/trt")
            
            from torch2trt import TRTModule
            from torch2trt import torch2trt
            
            h, w = map(int, args.eval_res.split(','))            
            
            model_path = "./weights/" + args.model + "_" + str(h) + "x" + str(w) + "_fp16.pth"
            if os.path.isfile(model_path):
                print("Loading weights from {}".format(model_path))
                model = TRTModule()
                model.load_state_dict(torch.load(model_path))
            else:        
                print("No existing model found. Converting and saving TRT model...")
                model = FASSDNet(19).cuda()
                state = convert_state_dict(torch.load(model_path_non_trt)["model_state"])    
                model.load_state_dict(state, strict=False)                
                model.eval()
                
                x = torch.rand((1, 3, int(h), int(w))).cuda()
                model = torch2trt(model, [x], fp16_mode=True)                
                
                weights_path = "./weights/" + args.model + "_" + str(h) + "x" + str(w) + "_fp16.pth"
                torch.save(model.state_dict(), weights_path)
                print("Done!")
        else:
            print("Error: TensorRT can only be used for a fixed input size through all the evaluation process...")
            raise SystemExit(0)
    else:
        print("No valid model found...")
                   
    inference(args, model, model_path)

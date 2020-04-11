import torch
from ptflops import get_model_complexity_info
from ptsemseg.models.FASSDNet import FASSDNet

with torch.cuda.device(0):
    net = FASSDNet(19)
    flops, params = get_model_complexity_info(net, (3, 512, 1024), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)


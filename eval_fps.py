import time
import torch
import torch.backends.cudnn as cudnn

from argparse import ArgumentParser
# from ptsemseg.models.FASSDNet import FASSDNet
# from ptsemseg.models.FASSDNetL1 import FASSDNet
# from ptsemseg.models.FASSDNetL2 import FASSDNet

def compute_speed(model, input_size, device, iteration=1000):
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()
    
    input = torch.randn(*input_size, device=device)
    
    for _ in range(50):
        model(input)
    
    print('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
        print("iteration {}/{}".format(_, iteration), end='\r')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("size", type=str, default="1024,2048", help="input size of model")
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--classes', type=int, default=19)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument("--alpha", default=2, nargs="?", type=int)  # NEW ALPHA IMPLEMENTATION
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
    
    parser.add_argument("--model", nargs="?", type=str,
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

    h, w = map(int, args.size.split(','))

    model = FASSDNet(args.classes, alpha=args.alpha)
    total_params = sum(p.numel() for p in model.parameters())
    print( 'Parameters:',total_params )

    compute_speed(model, (args.batch_size, args.num_channels, h, w), int(args.gpus), iteration=args.iter)

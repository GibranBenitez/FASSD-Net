# FASSD-Net: Fast and Accurate Semantic Segmentation with Dilated Asymmetric Convolutions

PyTorch implementation of our paper Fast and Accurate Semantic Segmentation with Dilated Asymmetric Convolutions, codes and pretrained models. 


Training was done using 2x NVIDIA TITAN RTX GPUs and Python 3.6.9. 

Speed was measured on a single NVIDIA GTX 1080ti GPU and an Intel Core i7-7700K 4.20GHz processor.


### Requirements
* pytorch 1.0.0
* torchvision 0.2.0
* scipy
* tqdm
* tensorboardX
* pill

### Usage

Change your Cityscapes path in the config files in order to run the training/validation scripts.

**To train the model :**

```
python train.py --model FASSDNet --config ./configs/FASSD-Net.yml
python train.py --model FASSDNetL1 --config ./configs/FASSD-Net-L1.yml
python train.py --model FASSDNetL2 --config ./configs/FASSD-Net-L2.yml

alternatively:
python train_nohup.py --model FASSDNet --config ./configs/FASSD-Net.yml
```

**To validate the model :**

```
python validate.py --model FASSDNet --model_path ./FASSD-Net-Cityscapes.pkl
python validate.py --model FASSDNetL1 --model_path ./FASSD-Net-L1-Cityscapes.pkl
python validate.py --model FASSDNetL2 --model_path ./FASSD-Net-L2-Cityscapes.pkl
```

**To measure the speed of the model :**

```
python eval_fps.py 1024,2048 --model FASSDNet
python eval_fps.py 1024,2048 --model FASSDNetL1
python eval_fps.py 1024,2048 --model FASSDNetL2
```

**Acknowledgement**

We thank PingoLH for releasing the [FCHarDNet](https://github.com/PingoLH/FCHarDNet) repo, which we build our work on top.

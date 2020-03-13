# FASSD-Net: Fast and Accurate Semantic Segmentation with Dilated Asymmetric Convolutions

PyTorch implementation of our paper Fast and Accurate Semantic Segmentation with Dilated Asymmetric Convolutions, codes and pretrained models.


### Requirements

* pytorch >=0.4.0
* torchvision ==0.2.0
* scipy
* tqdm
* tensorboardX

### Usage

Change your cityscapes path in the config files in order to run the training/validation scripts.

**To train the model :**

```
python train.py --model FASSDNet --config ./configs/FASSD-Net.yml
python train.py --model FASSDNetL1 --config ./configs/FASSD-Net-L1.yml
python train.py --model FASSDNetL2 --config ./configs/FASSD-Net-L2.yml
```

**To validate the model :**

```
python validate.py --model FASSDNet --model_path ./FASSD-Net-Cytiscapes.pkl
python validate.py --model FASSDNetL1 --model_path ./FASSD-Net-L1-Cytiscapes.pkl
python validate.py --model FASSDNetL2 --model_path ./FASSD-Net-L2-Cytiscapes.pkl
```

**To measure the speed of the model :**

```
python eval_fps.py 1024,2048 --model FASSDNet
python eval_fps.py 1024,2048 --model FASSDNetL1
python eval_fps.py 1024,2048 --model FASSDNetL2
```


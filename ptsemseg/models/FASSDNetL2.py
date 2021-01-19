import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


### --- Dilated Asymmetric Pyramidal Fusion module (DAPF)--- ###
class PyramBranch(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(PyramBranch, self).__init__()
        if padding == 0:
            print("Not supported for conv 1x1")

        else:
            self.atrous_conv3x1 = nn.Conv2d(inplanes, planes, kernel_size=(kernel_size, 1),
                                            stride=1, padding=(dilation, 0), dilation=(dilation, 1), bias=False)

            self.atrous_conv1x3 = nn.Conv2d(planes, planes, kernel_size=(1, kernel_size),
                                            stride=1, padding=(0, dilation), dilation=(1, dilation), bias=False)

            self.bn3x1 = BatchNorm(planes)
            self.relu3x1 = nn.ReLU()

            self.bn1x3 = BatchNorm(planes)
            self.relu1x3 = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv3x1(x)
        x = self.bn3x1(x)
        x = self.relu3x1(x)

        x = self.atrous_conv1x3(x)
        x = self.bn1x3(x)

        return self.relu1x3(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DAPF(nn.Module):
    def __init__(self, inplanes, alpha, BatchNorm):
        super(DAPF, self).__init__()

        dilations = [1, 12, 24, 36]
        mid_planes = inplanes//alpha

        self.conv1x1 = nn.Conv2d(inplanes, mid_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1x1 = BatchNorm(mid_planes)
        self.relu1x1 = nn.ReLU()

        self.pyBranch2 = PyramBranch(inplanes, mid_planes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.pyBranch3 = PyramBranch(inplanes, mid_planes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.pyBranch4 = PyramBranch(inplanes, mid_planes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.conv1 = nn.Conv2d(mid_planes*4, inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x1 = self.conv1x1(x)
        x1 = self.bn1x1(x1)
        x1 = self.relu1x1(x1)

        x2 = self.pyBranch2(x)
        x3 = self.pyBranch3(x)
        x4 = self.pyBranch4(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_DAPF(inplanes, alpha, BatchNorm):
    return DAPF(inplanes, alpha, BatchNorm)


### --- Muti-resolution Dilated Asymmetric module (MDA)--- ###
class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output

class MDA(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.parallel_conv3x3 = Conv(nIn // 2, nIn // 2, dkSize, 1,
                             padding=1, bn_acti=True)

        self.parallel_ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                              padding=(1 * d, 0), dilation=(d, 1), bn_acti=True)
        self.parallel_ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                              padding=(0, 1 * d), dilation=(1, d), bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.parallel_conv3x3(output)

        br2 = self.parallel_ddconv3x1(output)
        br2 = self.parallel_ddconv1x3(br2)

        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


### --- HarDBlock (HDB)--- ###
class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=kernel//2, bias = False))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)

    def forward(self, x):
        return super().forward(x)
        


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels
 
    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          layers_.append(ConvLayer(inch, outch))
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)


    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out



class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
    def forward(self, x, skip, concat=True):
        out = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True,
                            )
        if concat:                            
          out = torch.cat([out, skip], 1)
          
        return out

class FASSDNet(nn.Module):
    def __init__(self, n_classes=19, alpha=2):
        super(FASSDNet, self).__init__()

        first_ch  = [16, 24, 32, 48]
        ch_list = [64, 96, 160, 224, 320]
        grmul = 1.7
        gr       = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8]

        #### --- Encoder ---###
        blks = len(n_layers) 
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.base.append(ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2) )
        self.base.append(ConvLayer(first_ch[0], first_ch[1],  kernel=3, stride=2))
        self.base.append(ConvLayer(first_ch[1], first_ch[2],  kernel=3, stride=2))
        self.base.append(ConvLayer(first_ch[2], first_ch[3],  kernel=3))

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append (blk)
            if i < blks-1:
              self.shortcut_layers.append(len(self.base)-1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            
            if i < blks-1:            
              self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks-1
        self.n_blocks =  n_blocks

        # --- DAPF
        self.DAPF = build_DAPF(inplanes = prev_block_channels, alpha=alpha, BatchNorm=nn.BatchNorm2d)

        #### --- Decoder ---###
        dilation_block = [2, 4, 8, 16]

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up    = nn.ModuleList([])
        self.mda  = nn.ModuleList([])        

        for i in range(n_blocks-1,-1,-1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count//2, kernel=1))
            self.mda.append(MDA(cur_channels_count//2, d=dilation_block[i]))
            cur_channels_count = cur_channels_count//2

            blk = ConvLayer(cur_channels_count, 64)
            
            self.denseBlocksUp.append(blk)
            prev_block_channels = 64
            cur_channels_count = prev_block_channels


        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
               padding=0, bias=True)
               

    def forward(self, x):
        
        skip_connections = []
        size_in = x.size()
        
        
        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x
        
        out = self.DAPF(out)

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.mda[i](out)

            out = self.denseBlocksUp[i](out)
        
        out = self.finalConv(out)
        
        out = F.interpolate(
                            out,
                            size=(size_in[2], size_in[3]),
                            mode="bilinear",
                            align_corners=True)
        return out





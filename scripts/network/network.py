import torch
import torch.nn as nn
from torch import optim

import numpy as np

import MSA
import PDB

#CROP_LENGTH = 32  # needs to be even

def dataloader_2d(msa,il,ih,jl,jh):
    inputs = torch.zeros((483, ih-il,jh-jl))

    inputs[:20]   = torch.from_numpy(msa.hot[0][il:ih]).T.unsqueeze(2)
    inputs[20:40] = torch.from_numpy(msa.freq[il:ih]).T.unsqueeze(2)
    inputs[40:60] = torch.from_numpy(msa.hot[0][jl:jh]).T.unsqueeze(1)
    inputs[60:80] = torch.from_numpy(msa.freq[jl:jh]).T.unsqueeze(1)

    dca = msa.dca
    (a,b,c,d) = dca.shape
    dca = np.reshape(dca, (a,b,c*d))

    inputs[80:480] = torch.from_numpy(np.transpose(dca[il:ih,jl:jh],axes=(2,0,1)))
    inputs[480]    = torch.from_numpy(msa.apc[il:ih,jl:jh])
    inputs[481]    = torch.arange(il,ih).unsqueeze(1)
    inputs[482]    = torch.arange(jl,jh).unsqueeze(0)

    return inputs.unsqueeze(0)

def dataloader(msa, domain, i=0, j=0, k=0, crop_length=32, dev="cpu"):
    half = crop_length//2
    N = len(msa.seqs[0])

    il = max(0, i - half)
    jl = max(0, j - half)
    kl = max(0, k - half)

    ih = min(N, i + half)
    jh = min(N, j + half)
    kh = min(N, k + half)

    vil = il - i + half
    vjl = jl - j + half
    vkl = kl - k + half

    vih = ih - i + half
    vjh = jh - j + half
    vkh = kh - k + half

    inputs = dict()
    inputs["ijk"] = torch.zeros((1,1000, crop_length, crop_length, crop_length), device=dev)
    inputs["ijk"][0,:,vil:vih,vjl:vjh,vkl:vkh] = torch.from_numpy(msa.hodca(il,ih,jl,jh,kl,kh))
    inputs["jk"] = torch.zeros((1,483, crop_length, crop_length),device=dev)
    inputs["ik"] = torch.zeros((1,483, crop_length, crop_length),device=dev)
    inputs["ij"] = torch.zeros((1,483, crop_length, crop_length),device=dev)

    inputs["jk"][0,:,vjl:vjh,vkl:vkh] = dataloader_2d(msa,jl,jh,kl,kh)
    inputs["ik"][0,:,vil:vih,vkl:vkh] = dataloader_2d(msa,il,ih,kl,kh)
    inputs["ij"][0,:,vil:vih,vjl:vjh] = dataloader_2d(msa,il,ih,jl,jh)

    label = PDB.Label(domain, il,ih,jl,jh,kl,kh)
    l = torch.zeros((7,1,crop_length, crop_length, crop_length), dtype=torch.long, device=dev)
    view = l[:,0, vil:vih, vjl:vjh, vkl:vkh]
    view[0] = torch.from_numpy(label.dist).unsqueeze(1)
    view[1] = torch.from_numpy(label.alpha).unsqueeze(1)
    view[2] = torch.from_numpy(label.beta).unsqueeze(1)
    view[3] = torch.from_numpy(label.gamma).unsqueeze(1)
    view[4] = torch.from_numpy(label.theta)
    view[5] = torch.from_numpy(label.phi)
    view[6] = torch.from_numpy(label.psi)

    labels = dict()
    labels["dist"] =  l[0]
    labels["alpha"] = l[1]
    labels["beta"] =  l[2]
    labels["gamma"] = l[3]
    labels["theta"] = l[4]
    labels["phi"] =   l[5]
    labels["psi"] =   l[6]
    return inputs, labels

class Block2d(nn.Module):
    def __init__(self, channels, dilation=1):
        super(Block2d, self).__init__()
        self.sequential = self._make_sequential(channels, dilation)

    @staticmethod
    def _make_sequential(channels, dilation):
        layers = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels//4, kernel_size=1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(),
            nn.Conv2d(channels//4, channels//4, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm2d(channels//4),
            nn.Conv2d(channels//4, channels, kernel_size=1)
        )
        return layers

    def forward(self, x):
        return self.sequential(x) + x

class Block3d(nn.Module):
    def __init__(self, channels, dilation=1):
        super(Block3d, self).__init__()
        self.sequential = self._make_sequential(channels, dilation)

    @staticmethod
    def _make_sequential(channels, dilation):
        layers = nn.Sequential(
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels//4, kernel_size=1),
            nn.BatchNorm3d(channels//4),
            nn.ReLU(),
            nn.Conv3d(channels//4, channels//4, kernel_size=3, dilation=dilation, padding=dilation),
            nn.BatchNorm3d(channels//4),
            nn.Conv3d(channels//4, channels, kernel_size=1)
        )
        return layers

    def forward(self, x):
        return self.sequential(x) + x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.head2d = self._make_head2d()
        self.head3d = self._make_head3d()
        self.main = self._make_main()
        self.tail = self._make_tail()

    @staticmethod
    def _make_head2d():
        layers = [nn.Conv2d(483, 256, kernel_size=1)]
        for _ in range(2):
            for i in range(3):
                layers.append(Block2d(256, dilation=2**i))
        layers.append(nn.Conv2d(256, 128, kernel_size=1))
        for _ in range(5):
            for i in range(3):
                layers.append(Block2d(128, dilation=2**i))
        layers.append(nn.Conv2d(128, 64, kernel_size=1))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_head3d():
        layers = [nn.Conv3d(1000, 512, kernel_size=1)]
        for _ in range(2):
            for i in range(3):
                layers.append(Block3d(512, dilation=2**i))
        layers.append(nn.Conv3d(512, 256, kernel_size=1))
        for _ in range(5):
            for i in range(3):
                layers.append(Block3d(256, dilation=2**i))
        layers.append(nn.Conv3d(256, 128, kernel_size=1))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_main():
        layers = [nn.Conv3d(320, 256, kernel_size=1)]
        for _ in range(5):
            for i in range(3):
                layers.append(Block3d(256, dilation=2**i))
        layers.append(nn.Conv3d(256, 128, kernel_size=1))
        for _ in range(21):
            for i in range(3):
                layers.append(Block3d(128, dilation=2**i))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_tail():
        layers = [nn.BatchNorm3d(128),
            nn.Conv3d(128, 226, kernel_size=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        (a,b,c,d,e) = x["ijk"].shape
        jk = self.head2d(x["jk"]).unsqueeze(2).repeat(1,1,c,1,1)
        ik = self.head2d(x["ik"]).unsqueeze(3).repeat(1,1,1,d,1)
        ij = self.head2d(x["ij"]).unsqueeze(4).repeat(1,1,1,1,e)
        ijk = self.head3d(x["ijk"])

        x = torch.cat((jk,ik,ij,ijk), dim=1)

        x = self.main(x)
        x = self.tail(x)

        return self.out_dict(x)

    # This should probably be in the PDB class
    @staticmethod
    def out_dict(x):
        pred = dict()
        pred["dist"] = x[:,0:40]
        pred["alpha"] = x[:,40:59]
        pred["beta"] = x[:,59:96]
        pred["gamma"] = x[:,96:133]
        pred["theta"] = x[:,133:152]
        pred["phi"] = x[:,152:189]
        pred["psi"] = x[:,189:226]
        return pred

    def num_param(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

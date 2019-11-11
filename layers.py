# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np


'''custom layers
'''
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
    def __repr__(self):
        return self.__class__.__name__


class ConcatTable(nn.Module):
    '''ConcatTable container in Torch7.
    '''
    def __init__(self, layer1, layer2):
        super(ConcatTable, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        
    def forward(self,x):
        return [self.layer1(x), self.layer2(x)]


class Identity(nn.Module):
    '''
    nn.Identity in Torch7.
    '''
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    def __repr__(self):
        return self.__class__.__name__ + ' (skip connection)'


class Reshape(nn.Module):
    '''
    nn.Reshape in Torch7.
    '''
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(self.shape)
    def __repr__(self):
        return self.__class__.__name__ + ' (reshape to size: {})'.format(" ".join(str(x) for x in self.shape))


class CMul(nn.Module):
    '''
    nn.CMul in Torch7.
    '''
    def __init__(self):
        super(CMul, self).__init__()
    def forward(self, x):
        return x[0]*x[1]
    def __repr__(self):
        return self.__class__.__name__


class WeightedSum2d(nn.Module):
    def __init__(self):
        super(WeightedSum2d, self).__init__()
    def forward(self, x):
        x, weights = x
        assert x.size(2) == weights.size(2) and x.size(3) == weights.size(3),\
                'err: h, w of tensors x({}) and weights({}) must be the same.'\
                .format(x.size, weights.size)
        y = x * weights                                       # element-wise multiplication
        y = y.view(-1, x.size(1), x.size(2) * x.size(3))      # b x c x hw
        return torch.sum(y, dim=2).view(-1, x.size(1), 1, 1)  # b x c x 1 x 1
    def __repr__(self):
        return self.__class__.__name__


class SpatialAttention2d(nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    <!!!> attention score normalization will be added for experiment.
    '''
    def __init__(self, in_c, act_fn='relu'):
        super(SpatialAttention2d, self).__init__()
        self.act0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_c, 126, 1, 1)                 # 1x1 conv
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU(inplace=True)
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(126, 1, 1, 1)                    # 1x1 conv
        self.softplus = nn.Softplus(beta=1, threshold=20)       # use default setting.

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        s : softplus attention score 
        '''
        x = self.act0(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.softplus(x)
        return x
    
    def __repr__(self):
        return self.__class__.__name__

class ChannelAttention(nn.Module):
    def __init__(self, in_planes,actv ,ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        print('the ratio is ')
        print(ratio)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        if actv in ['softplus']:
            self.actv = nn.Softplus(beta=1, threshold=20)
        elif actv in ['relu']:
            self.actv = nn.ReLU()
        elif actv in ['sigmoid']:
            self.actv = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(x)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.actv(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class WPCA(nn.Module):
    def __init__(self,vlad_dim,nPCs,subtractMean= True,white = True):
        super(WPCA,self).__init__()
        self.vlad_dim=vlad_dim
        self.nPCs = nPCs
        self.subtractMean = subtractMean
        self.doWhite = white
        self.pca_layer = nn.Linear(vlad_dim,nPCs)
        print('pca vlad dim and npcs')
        print(vlad_dim, nPCs)

    def init_params(self,dataset):
        print('redaing pca param from file')
        with h5py.File('./pca_'+dataset+'_'+str(self.nPCs)+'_layer_params.hdf5',mode='r') as h5:
            U = torch.Tensor(h5['U'][:])
            lams = torch.Tensor(h5['lams'][:])
            mu= torch.Tensor(h5['xm'][:])
            Utmu = torch.Tensor(h5['Utmu'][:])
        self.pca_layer.weight = nn.Parameter(torch.t(U))
        self.pca_layer.bias = nn.Parameter(torch.t(-Utmu))
        print('got weight and bias size')
        print(self.pca_layer.weight.size(),self.pca_layer.bias.size())

    def calc_dbfeats(self,dataloader,whole_training_data_loader,model):
        print('calculating deFeats')
        with h5py.File('./vgg16_dbFeats.hdf5',mode='w') as h5:
            h5feat = h5.create_dataset("features", 
                    [len(dataloader), self.vlad_dim], 
                        dtype=np.float32)
            for iteration, (input, indices) in enumerate(whole_training_data_loader, 1):
                input = input.to(device='cuda',dtype=torch.float)
                vlad_encoding = model(input).cuda()
                h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
        print('extract dbfeats finished')
        assert(False)
    def calc_pca_params(self,arch):
        with h5py.File('./vgg16_dbFeats.hdf5',mode='r') as h5:
            h5feat = h5['features'][:]
            #a = h5feat[:]
        U, lams, xm, Utmu = pca(h5feat.T,self.nPCs)
        with h5py.File('./pca_'+arch+'_'+str(self.nPCs)+'_layer_params.hdf5', mode='w') as h5:
            h5.create_dataset('U',data = U.cpu().numpy())
            h5.create_dataset('Utmu',data = Utmu.cpu().numpy())
            h5.create_dataset('lams', data = lams.cpu().numpy())
            h5.create_dataset('xm',data = xm.cpu().numpy())
        print('finish pca calculating')
        assert(False)
    def forward(self,x):
        #print(x.size())
        x = self.pca_layer(x)
        #print(x.size())
        x = F.normalize(x, p=2, dim=1)
        return x
class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out

class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
#        print(feature.size())
#        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)
class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        #print(feature_A.size())
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w).detach().cpu().numpy()
        feature_B = feature_B.view(b,c,h*w).transpose(1,2).detach().cpu().numpy()
        # perform matrix mult.
        #torch.cuda.synchronize()
        #feature_mul = torch.matmul(feature_B,feature_A)
       # print(feature_B.shape,feature_A.shape)
        feature_mul = np.matmul(feature_B,feature_A)
        feature_mul = torch.from_numpy(feature_mul).cuda()
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        #print(correlation_tensor.size())
        return correlation_tensor
    
class FeatureRegression(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(961, 128, kernel_size=7, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.linear = nn.Linear(64 * 21 * 21, output_dim)
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        x = self.linear(x)
        return x
class HeatmapRegression(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True):
        super(HeatmapRegression, self).__init__()

        self.linear = nn.Linear(31*31, output_dim)
        if use_cuda:
            self.linear.cuda()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

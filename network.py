import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb
import torch.nn.functional as F
import h5py
import layers


from layers import (
    CMul, 
    Flatten, 
    ConcatTable, 
    Identity, 
    Reshape, 
    SpatialAttention2d, 
    WeightedSum2d,
    ChannelAttention,
    SpatialAttention,
    WPCA,
    InstanceNormalization,
    FeatureRegression,
    FeatureCorrelation,
    FeatureL2Norm,
    HeatmapRegression
    )

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss

def pca(x,nPCs,subtractMean=True): 
    nDims,nPoints= x.shape
    x= torch.from_numpy(x).cuda()
    if subtractMean:
        xm = torch.t(torch.mean(x,1).unsqueeze(0))
        x = x - xm.expand_as(x)
    if nDims<=nPoints:  
        doDual = False
        x2 = torch.matmul(x,torch.t(x))/(nPoints-1)
    else:
        doDual = True
        x2 = torch.matmul(torch.t(x),x)/(nPoints-1)
    (e,v)=torch.eig(x2,eigenvectors=True)
    
    if nPCs<x2.size(0):
        lams = e[:nPCs,0]
        print(len(lams))
        L = torch.diag(e[:nPCs,0], diagonal=0, out=None)
        U = v[:,:nPCs]
    else:
        
        lams = e[:,0]
        print(len(lams))
        L = torch.diag(e[:,0], diagonal=0, out=None)
        U = v
    sortInd = torch.sort(lams,descending=True)[1]
    U= U[:,sortInd]
    if doDual:
        lams[np.where(lams<0)]=0
        m = torch.matmul(U,torch.diag(1./np.sqrt(lams)))
        U = torch.matmul(x,m)/np.sqrt(nPoints-1)
    Utmu = torch.matmul(torch.t(U),xm)

    return U, lams, xm, Utmu



class EmbedNet2(nn.Module):
    def __init__(self, net_vlad,num_cluster,dim,PCA_dim,pca_module,pretrained=True):
        super(EmbedNet2, self).__init__()

        encoder = models.alexnet(pretrained=pretrained)
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            for l in layers[:-2]:
                for p in l.parameters():
                    p.requires_grad = False

        self.base_model = nn.Sequential(*layers)
        
        self.net_vlad = net_vlad
        self.do_pca=False
       # if PCA_dim>0 and pca_module is not None:
       #     self.do_pca=True
       #     self.WPCA = pca_module
       #     print(pca_module)
       #     print('---------------using pca-------------')

    def forward(self, x):
        x = self.base_model(x)
        x = self.net_vlad(x)
        #print(x.size())
        #print(type(x))
        #if self.do_pca:
         #   x = self.WPCA(x)
        return x
class AttenPoolingAlex(nn.Module):
    def __init__(self,net_vlad,num_cluster,dim,relu,mode,add_relu=False,attention=None,freeze=True,pretrained=True):
        super(AttenPoolingAlex, self).__init__()
        self.mode = mode
        encoder = models.alexnet(pretrained=pretrained)
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            if freeze:
                num_freeze = -1
            else:
                num_freeze = -3
                print('train conv 4 and 5')
            for l in layers[:num_freeze]:
                for p in l.parameters():
                    p.requires_grad = False

        self.base_model = nn.Sequential(*layers)

        self.net_vlad = net_vlad
        print(attention)
        self.relu1=None
        if attention in ['delf_attention']:
            self.atten = SpatialAttention2d(dim)
            print('using delf attention')
            relu=None
        else:
            a_layer=[]
            if add_relu:
                a_layer.append(nn.ReLU())
            a_layer.append(nn.Conv2d(dim,1,1,1))
            print('using attention to get heatmap')
            if relu == 'relu':
                a_layer.append(nn.ReLU())
                print('using relu')
            elif relu =='prelu':
                print('**using prelu')
                a_layer.append(nn.PReLU())
            elif relu =='sigmoid':
                print('using sigmoid')
                a_layer.append(nn.Sigmoid())
            elif relu =='softplus':
                print('using softplus')
                a_layer.append(nn.Softplus(beta=1, threshold=20))
            self.atten = nn.Sequential(*a_layer) #nn.Conv2d(dim,1,1,1,bias=False)#nn.Sequential(*a_layer)
    def forward(self, x):
        x = self.base_model(x)
        hmp = self.atten(x)
        x = x*hmp
        x = self.net_vlad(x)
        #x = F.normalize(x,p=2,dim=1)
        if self.mode in ['test']:
            out = (x,hmp)
        else:
            out = x
        return out

def load_base_module(arch,pretrain,freeze):
    encoder = models.__dict__[arch](pretrained=True)
    if arch in ['alexnet']:
        layers = list(encoder.features.children())[:-2]
        if pretrain:
            if freeze:
                num_freeze = -1
            else:
                num_freeze = -3
                print('train conv 4 and 5')
            for l in layers[:num_freeze]:
                for p in l.parameters():
                    p.requires_grad = False
        else:
            print('train all layers')
    elif arch in ['vgg16']:
        layers = list(encoder.features.children())[:-2]
        if pretrain:
            print('load pretrained vgg16 and only train conv5') # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False
        else:
            print('train all vgg16 layers')
    base_model = nn.Sequential(*layers)
    return base_model

class AttenVLADNet(nn.Module):
    def __init__(self,net_vlad,arch,mode,freeze=True,Pretrained=True):
        super(AttenVLADNet, self).__init__()
        self.mode = mode
        self.base_model = load_base_module(arch,Pretrained,freeze)
        self.net_vlad = net_vlad
    def forward(self, x):
        x = self.base_model(x)
        x,hmp = self.net_vlad(x)
        if self.mode in ['test']:
            out = (x,hmp)
        else:
            out = x
        return out

def load_alexnet(Pretrained,freeze):
    encoder = models.alexnet(pretrained=True)
    layers = list(encoder.features.children())[:-2]
    if Pretrained:
        if freeze:
            num_freeze = -1
        else:
            num_freeze = -3
            print('train conv 4 and 5')
        for l in layers[:num_freeze]:
            for p in l.parameters():
                p.requires_grad = False
    else:
        print('train all layers')
    model = nn.Sequential(*layers)
    return model
class AttenVLADAlex(nn.Module):
    def __init__(self,net_vlad,num_cluster,dim,relu,mode,freeze=True,Pretrained=True,pca=None,da_type=None):
        super(AttenVLADAlex, self).__init__()
        #self.mode = mode
        self.da_type = da_type
        self.base_model = load_alexnet(Pretrained,freeze)
        self.net_vlad = net_vlad
        self.pca=False
        if pca is not None:
            print('using pca module')
            self.pca = True
            self.WPCA = pca
        if da_type in ['f_affine']:
            print('========>using f affine')
            self.FeatureCorrelation = FeatureCorrelation()
            self.FeatureRegression = FeatureRegression()

        self.affine_match = HeatmapRegression()
        self.FeatureL2Norm = FeatureL2Norm()
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.base_model(x)
        #x = F.normalize(x, p=2, dim=1)
        if self.da_type in ['coral','affine']:
            x,hmp = self.net_vlad(x)
            hmp = self.FeatureL2Norm(self.ReLU(hmp))
            x1 = self.affine_match(hmp)
        elif self.da_type in ['f_affine']:
            x,hmp = self.net_vlad(x)
            correlation = self.FeatureCorrelation(hmp[:2],hmp[2:])
            correlation = self.FeatureL2Norm(self.ReLU(correlation))
            x1 = self.FeatureRegression(correlation)
            print(x1.size())
            print('x1size')
        else:
            x = self.net_vlad(x)
        if self.pca:
            x = self.WPCA(x)

        if self.da_type in ['coral','affine','f_affine']:
            x = (x,x1)

        return x

class RecAttentionAlex(nn.Module):
    def __init__(self,net_vlad,num_clusters,dim,relu,mode,add_relu=False,attention=None,freeze=True,rect_atten=True,pca_dim=-1,pretrained=True):
        super(RecAttentionAlex, self).__init__()
        self.mode = mode
        self.dopca = pca_dim >0
        if self.dopca:
            print('pca reduce to {}'.format(pca_dim))
            self.WPCA = WPCA(num_clusters*dim,pca_dim)
            self.WPCA.init_params('mapillary')

        encoder = models.alexnet(pretrained=pretrained)
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            if freeze:
                num_freeze = -1
            else:
                num_freeze = -3
                print('train conv 4 and 5')
            for l in layers[:num_freeze]:
                for p in l.parameters():
                    p.requires_grad = False

        self.base_model = nn.Sequential(*layers)
        self.net_vlad = net_vlad


        if attention in ['delf_attention']:
            self.atten = SpatialAttention2d(dim)
            print('using delf attention')
        else:
            a_layer=[]
            if add_relu:
                a_layer.append(nn.ReLU())
            a_layer.append(nn.Conv2d(dim,1,1,1))

            print('using attention jj to get heatmap')
            if relu == 'relu':
                a_layer.append(nn.ReLU())
                print('using relu')
            elif relu =='prelu':
                print('**using prelu')
                a_layer.append(nn.PReLU())
            elif relu =='sigmoid':
                print('using sigmoid')
                a_layer.append(nn.Sigmoid())
            elif relu =='softplus':
                print('using softplus')
                a_layer.append(nn.Softplus(beta=1, threshold=20))
            self.atten = nn.Sequential(*a_layer) #nn.Conv2d(dim,1,1,1,bias=False)#nn.Sequential(*a_layer)
        #self.ca = ChannelAttention(dim)
        self.rect_atten = rect_atten
        self.num_clusters = num_clusters
 
    def forward(self, x):
        x = self.base_model(x)
        hmp = self.atten(x)
        if self.rect_atten:
            N, C = x.shape[:2]
            Nmp,Cmp = hmp.shape[:2]
            res = self.net_vlad(x)
            vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device) # 24 64 256
            
            hmp_flatten = hmp.view(Nmp,Cmp,-1)  # 24,1,961
            for C in range(self.num_clusters):
                residual = res[:,C:C+1,:,:]*hmp_flatten.unsqueeze(2) # 24 1 256 961 *24 1 1 961 
                vlad[:,C:C+1,:] = residual.sum(dim=-1) # 24 1 256
            vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
            vlad = vlad.view(x.size(0), -1)  # flatten
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize #[24, 16384]
            x = vlad
        else:
            print('raw simple mutiply with attention map')
            x = x*hmp
            out = self.net_vlad(x)
        if self.dopca:
            x = self.WPCA(x)

        if self.mode in ['test']:
            out = (x,hmp)
        else:
            out = x

        return out

class AttenNet(nn.Module):
    def __init__(self,net_vlad,arch,num_cluster,dim,relu,mode,freeze=False):
        super(AttenPoolingAlex, self).__init__()
        self.mode = mode
        
        encoder = models.alexnet(pretrained=pretrained)
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            if freeze:
                num_freeze = -1
            else:
                num_freeze = -3
                print('train conv 4 and 5')
            for l in layers[:num_freeze]:
                for p in l.parameters():
                    p.requires_grad = False

        self.base_model = nn.Sequential(*layers)
        self.WPCA = pca_model
        self.net_vlad = net_vlad
        self.atten = nn.Conv2d(dim,1,kernel_size=1)#,bias=False)
        print('using attention to get heatmap')
        self.atten.weight.data.normal_(0, 0.01)
        if self.atten.bias is not None:
            self.atten.bias.data.fill_(0)
        self.relu1=None
        if relu == 'relu':
            self.relu1=nn.ReLU()
            print('using relu')
        elif relu =='prelu':
            print('**using prelu')
            self.relu1=nn.PReLU()
        elif relu =='sigmoid':
            print('using sigmoid')
            self.relu1 = nn.Sigmoid()
        elif relu =='tanh':
            print('using tanh')
            self.relu1 = nn.Tanh()
        print(relu)
    def forward(self, x):
        x = self.base_model(x)
        if self.relu1:
            hmp = self.relu1(self.atten(x))
        else:
            hmp = self.atten(x)
        x = x*hmp
        x = self.net_vlad(x)
        x = F.normalize(x,p=2,dim=1)
        if self.mode in ['test']:
            out = (x,hmp)
        else:
            out = x
        return out

class AttentAlex(nn.Module):
    def __init__(self,net_vlad,num_cluster,dim,PCA_dim,mode=None,BatchNorm=None,pca_model=None,pretrained=True):
        super(AttentAlex, self).__init__()
        self.doPCA = pca_model
        self.mode=mode
        encoder = models.alexnet(pretrained=pretrained)
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False
        layers.append(nn.BatchNorm2d(dim))
        self.base_model = nn.Sequential(*layers)
        self.WPCA = pca_model
        self.net_vlad = net_vlad

        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.base_model(x)
        x = self.ca(x)*x
        hmp = self.sa(x)
        x = hmp*x
        x = self.net_vlad(x)
        if self.doPCA:
            assert(self.WPCA)
            x = self.WPCA(x)
        x = F.normalize(x,p=2,dim=1)
        if self.mode in ['test']:
            out = (x,hmp)
        else:
            out = x
        return x
class VGGNet(nn.Module):
    def __init__(self,net_vlad,num_cluster,dim,PCA_dim,BatchNorm=None,pca_model=None,pretrained=True):
        super(VGGNet, self).__init__()
        self.doPCA = pca_model
        encoder = models.vgg16(pretrained=pretrained)
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False
        layers.append(nn.BatchNorm2d(dim))
        self.base_model = nn.Sequential(*layers)
        self.WPCA = pca_model

#        self.ca = ChannelAttention(dim)
#        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.base_model(x)
        x = self.net_vlad(x)
#        x = self.ca(x)*x
#        x = self.sa(x)*x
        if self.doPCA:
            assert(self.WPCA)
            x = self.WPCA(x)
        x = F.normalize(x,p=2,dim=1)
        return x
class AttenPoolingVGG(nn.Module):
    def __init__(self,net_vlad,num_cluster,dim,PCA_dim,BatchNorm=None,pca_model=None,pretrained=True):
        super(AttenPoolingVGG, self).__init__()
        self.doPCA = pca_model
        self.mode=mode
        encoder = models.alexnet(pretrained=pretrained)
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False

        self.base_model = nn.Sequential(*layers)
        self.WPCA = pca_model
        self.net_vlad = net_vlad
        self.atten = nn.Conv2d(dim,1,kernel_size=1)
        print('using attention to get heatmap')
        self.atten.weight.data.normal_(0, 0.01)
        self.atten.bias.data.fill_(0)

    def forward(self, x):
        x = self.base_model(x)
        x = self.atten(x)*x
        x = self.net_vlad(x)
        if self.doPCA:
            assert(self.WPCA)
            x = self.WPCA(x)
        x = F.normalize(x,p=2,dim=1)
        if self.mode in ['test']:
            out = (x,hmp)
        else:
            out = x
        return x
class AttentVGGNet(nn.Module):
    def __init__(self,net_vlad,num_cluster,dim,PCA_dim,mode,BatchNorm=None,pca_model=None,pretrained=True):
        super(AttentVGGNet, self).__init__()
        self.doPCA = pca_model
        self.mode = mode
        encoder = models.vgg16(pretrained=pretrained)
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            for l in layers[:-5]:
                for p in l.parameters():
                    p.requires_grad = False
        self.base_model = nn.Sequential(*layers)
        self.WPCA = pca_model
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()

        self.net_vlad = net_vlad

    def forward(self, x):
        x = self.base_model(x)
        x = self.ca(x)*x
        hmp = self.sa(x)
        x = hmp*x
        x = self.net_vlad(x)
        x = F.normalize(x,p=2,dim=1)
        if self.mode in ['test']:
            out = (x,hmp)
            print('output heatmap')
        else:
            out = x
        return out
class MyNet(nn.Module):
    def __init__(self, net_vlad,num_cluster,dim,PCA_dim, BatchNorm=None):
        super(MyNet, self).__init__()
        self.do_pca = PCA_dim>0
 
        self.base_model = nn.Sequential(nn.Conv2d(1, 32, 3), 
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=3, stride=2),
                                        nn.Conv2d(32, 64, 3), 
            )
        if BatchNorm == 'bn_vlad':
            self.net_vlad = nn.Sequential(
                             net_vlad,
                             nn.BatchNorm1d(num_cluster*dim),
            )
        else:
            self.net_vlad = net_vlad
        if self.do_pca:
            print(self.do_pca)
            if BatchNorm == 'bn_pca':
                self.WPCA = nn.Sequential(
                nn.Linear(num_cluster*dim,PCA_dim),
                nn.BatchNorm1d(PCA_dim),
                       )
            else:
                print(num_cluster*dim,PCA_dim)
                self.WPCA = nn.Linear(num_cluster*dim,PCA_dim)
    def forward(self, x):
        x = self.base_model(x)
        x = self.net_vlad(x)
        if self.do_pca:
            x = self.WPCA(x)
            x = F.normalize(x,p=2,dim=1)
        return x

class MyLeNet(nn.Module):
    def __init__(self, net_vlad,num_cluster,dim,PCA_dim, BatchNorm=None):
        super(MyLeNet, self).__init__()
        self.do_pca = PCA_dim>0
 
        self.base_model = nn.Sequential(nn.Conv2d(1, 20, 5), 
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2),
                                        nn.Conv2d(20, 50, 5), 
            )
        if BatchNorm == 'bn_vlad':
            self.net_vlad = nn.Sequential(
                             net_vlad,
                             nn.BatchNorm1d(num_cluster*dim),
            )
        else:
            self.net_vlad = net_vlad
        if self.do_pca:
            print(self.do_pca)
            if BatchNorm == 'bn_pca':
                self.WPCA = nn.Sequential(
                nn.Linear(num_cluster*dim,PCA_dim),
                nn.BatchNorm1d(PCA_dim),
                       )
            else:
                print(num_cluster*dim,PCA_dim)
                self.WPCA = nn.Linear(num_cluster*dim,PCA_dim)
    def forward(self, x):
        x = self.base_model(x)
        x = self.net_vlad(x)
        if self.do_pca:
            x = self.WPCA(x)
            x = F.normalize(x,p=2,dim=1)
        return x

class DeepCORAL(nn.Module):
    def __init__(self, model):
        super(DeepCORAL, self).__init__()
        self.sharedNet = model

    def forward(self, source, target,cache=False):
        if cache:
            out = self.sharedNet(source)
        else:
            source = self.sharedNet(source)
            target = self.sharedNet(target)
            out = (source,target)  
        return out

class DeepCORAL2(nn.Module):
    def __init__(self, model):
        super(DeepCORAL2, self).__init__()
        self.sharedNet_base = model.base_model
        self.sharedNet_vlad = model.net_vlad
       # self.sharedNet = model

    def forward(self, source, target,cache=False,mode=None):
        if cache:
            out = self.sharedNet_base(source)
            out = self.sharedNet_vlad(out)
        else:
            if mode in ['feature']:
                source = self.sharedNet_base(source)
                target = self.sharedNet_base(target)
                out = (source,target) 
            elif mode in ['vlad']:
                out = self.sharedNet_vlad(source)
                #target = self.sharedNet_vlad(target)
            else:
                print('Error')
            
        return out

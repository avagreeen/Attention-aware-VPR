import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
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
    WPCA
    )
# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, 
                 normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            print('using vladv1')
            print(clsts,traindescs)
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            print(clstsAssign)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending
            print(np.mean(dots[0,:] - dots[1,:]),type(np.mean(dots[0,:] - dots[1,:])))
            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            print('using vladv2')
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                        self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class NetVLAD_res(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, 
                 normalize_input=True, vladv2=False):
        super(NetVLAD_res, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))



    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            print('using vladv1')
            print(clsts,traindescs)
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            print(clstsAssign)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending
            print(np.mean(dots[0,:] - dots[1,:]),type(np.mean(dots[0,:] - dots[1,:])))
            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            print('using vladv2')
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )
    def forward(self, x):
        N, C = x.shape[:2]   # 24 256 31 31

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)    # 24 64 961
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)   #24 256 961
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, device=x.device) #24 64 256
        residuals = torch.zeros([N, self.num_clusters, C ,961], dtype=x.dtype, layout=x.layout, device=x.device)  # 24 64 256 961
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)   # 24 1 256 961 * 24 1 1 961  = 24 1 256 961 
            #print(residual.size())
            #print('res size')
            residuals[:,C:C+1,:,:] = residual  # [24, 1, 256, 961]
        #print('return residuals {}'.format(residuals.size()))
        
        return residuals


class AttenNetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=128,attention=None,mul=True,actv='relu',da_type=None,add_relu=True,
                 normalize_input=True, vladv2=False,ratio=4):
        super(AttenNetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.da_type = da_type
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.mul = mul
        print(mul)
        print('***')
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.attention=attention
        if attention in ['casa']:
            self.ca = ChannelAttention(dim,actv,ratio)
            print('using channel attention')
        if attention in ['delf_attention']:
            self.atten = SpatialAttention2d(in_c=dim,act_fn=actv)
            print('using delf attention')
        #elif attention in ['casa']:
        #    self.ca = ChannelAttention(dim,actv,ratio)
        #   self.atten = ChannelSpatialAttention(dim)    
        else:
            a_layer=[]
            if add_relu:
                a_layer.append(nn.ReLU(inplace=True))
            a_layer.append(nn.Conv2d(dim,1,1,1))
            print('using attention to get heatmap')
            if actv == 'relu':
                a_layer.append(nn.ReLU())
                print('using relu')
            elif actv =='prelu':
                print('**using prelu')
                a_layer.append(nn.PReLU())
            elif actv =='sigmoid':
                print('using sigmoid')
                a_layer.append(nn.Sigmoid())
            elif actv =='softplus':
                print('using softplus')
                a_layer.append(nn.Softplus(beta=1, threshold=20))
            self.atten = nn.Sequential(*a_layer) #nn.Conv2d(dim,1,1,1,bias=False)#nn.Sequential(*a_layer)

    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            print('using vladv1')
            print(clsts,traindescs)
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            print(clstsAssign)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending
            print(np.mean(dots[0,:] - dots[1,:]),type(np.mean(dots[0,:] - dots[1,:])))
            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            print('using vladv2')
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def __vlad_compute__(self, x_flatten,hmp_flatten,soft_assign,N,C):
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x_flatten.dtype, device=x_flatten.device) #24 64 256
        #print('del the res')
        hmp_flatten = hmp_flatten.unsqueeze(2)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)   # 24 1 256 961 * 24 1 1 961  = 24 1 256 961
            residual *=  hmp_flatten
            vlad[:,C:C+1,:] = residual.sum(dim=-1)     #vlas.size = 24 64 256 961
        #del residuals
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad
    def __ori_vlad_compute__(self, x_flatten,soft_assign,N,C):
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x_flatten.dtype, device=x_flatten.device) #24 64 256
        #print('del the res')
        #hmp_flatten = hmp_flatten.unsqueeze(2)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)   # 24 1 256 961 * 24 1 1 961  = 24 1 256 961
            #residual *=  hmp_flatten
            vlad[:,C:C+1,:] = residual.sum(dim=-1)     #vlas.size = 24 64 256 961
        #del residuals
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def forward(self, x):
        if self.attention in ['casa']:
            x = x*self.ca(x)

        hmp = self.atten(x)
        if self.mul in ['mul']:     
            x = hmp*x

        N, C = x.shape[:2]   # 24 256 31 31
        Nmp,Cmp = hmp.shape[:2]
        hmp_flatten = hmp.view(Nmp,Cmp,-1) 

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        x_flatten = x.view(N, C, -1)   #24 256 961
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)    # 24 64 961
        soft_assign = F.softmax(soft_assign, dim=1)
        
        vlad = self.__vlad_compute__(x_flatten,hmp_flatten,soft_assign,N,C)
        del soft_assign
        if self.mul in ['2mul']:
            soft_assign = self.conv(x*hmp).view(N, self.num_clusters, -1)    # 24 64 961
            soft_assign = F.softmax(soft_assign, dim=1)
            x_mul = hmp_flatten*x_flatten
            mul_vlad = self.__vlad_compute__(x_mul,hmp_flatten,soft_assign,N,C)
            del soft_assign,x_mul
            vlad = mul_vlad*vlad
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        # calculate residuals to each clusters
        elif self.mul in ['3mul']:
            soft_assign = self.conv(x*hmp).view(N, self.num_clusters, -1)    # 24 64 961
            soft_assign = F.softmax(soft_assign, dim=1)
            x_mul = hmp_flatten*x_flatten
            mul_vlad = self.__vlad_compute__(x_mul,hmp_flatten,soft_assign,N,C)
            del soft_assign,x_mul
            vlad = mul_vlad+vlad
            vlad = F.normalize(vlad, p=2, dim=1) 
        elif self.mul in ['4mul']:
            soft_assign = self.conv(x*hmp).view(N, self.num_clusters, -1)    # 24 64 961
            soft_assign = F.softmax(soft_assign, dim=1)
            x_mul = hmp_flatten*x_flatten
            mul_vlad = self.__ori_vlad_compute__(x_mul,soft_assign,N,C)
            del soft_assign,x_mul
            vlad = mul_vlad+vlad
            vlad = F.normalize(vlad, p=2, dim=1) 
      #  if self.attention in ['casa']:
            #print('enforcing channel attention*')
            #print(vlad.size(),cweight.size())  # 24 256 1 1
       #     vlad *= cweight.squeeze(3).permute(0,2,1)
        if self.da_type in ['coral','affine']:
            vlad = (vlad,hmp[:4,...])
        elif self.da_type in ['f_affine']:
            x1 = x[:4,...]*hmp[:4,...]
            #print(x1.size(),x[:4,...].size(),hmp[:4,...].size())  torch.Size([4, 256, 31, 31]) torch.Size([4, 256, 31, 31]) torch.Size([4, 1, 31, 31])
            
            vlad = (vlad, x1)
        else:
            pass
        return vlad





def vlad_compute(x_flatten,hmp_flatten,num_clusters,centroids):
    vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, device=x.device) #24 64 256
    residuals = torch.zeros([N, self.num_clusters, C ,961], dtype=x.dtype, layout=x.layout, device=x.device)  # 24 64 256 961

    for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
        residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign[:,C:C+1,:].unsqueeze(2)   # 24 1 256 961 * 24 1 1 961  = 24 1 256 961
        residual *=  hmp_flatten.unsqueeze(2)
        vlad[:,C:C+1,:] = residual.sum(dim=-1)     #vlas.size = 24 64 256 961
    #if self.attention in ['casa']:
        #print('enforcing channel attention*')
        #print(vlad.size(),cweight.size())  # 24 256 1 1
    #    vlad *= cweight.squeeze(3).permute(0,2,1)
    vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
    vlad = vlad.view(x.size(0), -1)  # flatten
    vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
    return vlad

class Atten2NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=128,attention=None,mul=True,actv='relu',add_relu=True,
                 normalize_input=True, vladv2=False):
        super(Atten2NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.mul = mul
        print(mul)
        print('***')
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.attention=attention
        if attention in ['delf_attention']:
            self.atten = SpatialAttention2d(in_c=dim,act_fn=actv)
            print('using delf attention')
        elif attention in ['casa']:
            self.ca = ChannelAttention(dim,actv)
        else:
            a_layer=[]
            if add_relu:
                a_layer.append(nn.ReLU())
            a_layer.append(nn.Conv2d(dim,1,1,1))
            print('using attention to get heatmap')
            if actv == 'relu':
                a_layer.append(nn.ReLU())
                print('using relu')
            elif actv =='prelu':
                print('**using prelu')
                a_layer.append(nn.PReLU())
            elif actv =='sigmoid':
                print('using sigmoid')
                a_layer.append(nn.Sigmoid())
            elif actv =='softplus':
                print('using softplus')
                a_layer.append(nn.Softplus(beta=1, threshold=20))
            self.atten = nn.Sequential(*a_layer) #nn.Conv2d(dim,1,1,1,bias=False)#nn.Sequential(*a_layer)

    def init_params(self, clsts, traindescs):
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            print('using vladv1')
            print(clsts,traindescs)
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            print(clstsAssign)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending
            print(np.mean(dots[0,:] - dots[1,:]),type(np.mean(dots[0,:] - dots[1,:])))
            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            print('using vladv2')
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def __vlad_compute__(self, x_flatten,hmp_flatten,soft_assign,N,C):
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x_flatten.dtype, device=x_flatten.device) #24 64 256
        residuals = torch.zeros([N, self.num_clusters, C ,961], dtype=x_flatten.dtype,  device=x_flatten.device)  # 24 64 256 961

        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)   # 24 1 256 961 * 24 1 1 961  = 24 1 256 961
            residual *=  hmp_flatten.unsqueeze(2)
            vlad[:,C:C+1,:] = residual.sum(dim=-1)     #vlas.size = 24 64 256 961

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def forward(self, x):

        hmp = self.atten(x)

        N, C = x.shape[:2]   # 24 256 31 31
        Nmp,Cmp = hmp.shape[:2]
        hmp_flatten = hmp.view(Nmp,Cmp,-1) 

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        x_flatten = x.view(N, C, -1)   #24 256 961
        # soft-assignment　　　　　　　

        soft_assign = self.conv(x*hmp).view(N, self.num_clusters, -1)    # 24 64 961
        soft_assign = F.softmax(soft_assign, dim=1)
        x_mul = hmp_flatten*x_flatten
        mul_vlad = self.__vlad_compute__(x_mul,hmp_flatten,soft_assign,N,C)
        del soft_assign,x_mul
        vlad = mul_vlad
        vlad = F.normalize(vlad, p=2, dim=1) 

        return vlad


class MYPCA(nn.Module):
    def __init__(self, num_pca):
        super(MYPCA, self).__init__()
        self.num_pca = num_pca
    def forward(self,X):
        #X = torch.t(X)
        X_mean = torch.mean(X,0)
        X = X - X_mean.expand_as(X)
        print(X.size())
        print(self.num_pca)
        U,S,V = torch.svd(torch.t(X))
        C = torch.mm(X,U[:,:self.num_pca])
        print(C.size())
        return C

from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import faiss
import network
import os
from tensorboardX import SummaryWriter
import numpy as np
import netvlad
import myloss
import layers
import time
from collections import OrderedDict
parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster','fine-tune'])
parser.add_argument('--batchSize', type=int, default=24, 
        help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000, 
        help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
        help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=0, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=16, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='./data/', help='Path for centroid data.')
parser.add_argument('--runsPath', type=str, default='./runs/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='./checkpoints/', 
        help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default='./cache/', help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest', 
        help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=5, 
        help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
parser.add_argument('--dataset', type=str, default='mapillary',choices = ['pittsburgh','mapillary','mnist','m2u'], 
        help='Dataset to use')
parser.add_argument('--arch', type=str, default='self_define', 
        help='basenetwork to use', choices=['vgg16', 'alexnet','self_define'])
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use', 
        choices=['netvlad', 'max', 'avg','atten_vlad'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val', 
        choices=['test', 'test250k', 'train', 'val','beelbank'])
parser.add_argument('--fromscratch', action='store_true',default=False, help='Train from scratch rather than using pretrained models')
parser.add_argument('--pretrain', type=str, default='', help='Path to load pretrainde model from.')
parser.add_argument('--num_PCA', type=int, default = '-1', help='dimention for PCAW')
parser.add_argument('--bn', type=str, default='none', help='way of bn',choices=['relu', 'bn_pca', 'bn_vlad'])
parser.add_argument('--DA', action='store_true')
parser.add_argument('--random_pos_level', type=int,default=1, help='choose random positives')
parser.add_argument('--random_crop', action='store_true',default=False, help='choose random positives')
parser.add_argument('--casa', action='store_true',default=False, help='choose random positives')
parser.add_argument('--loss', type=str, default='l2', choices=['l2','cos','impr_triplet','mmd'] ,help='Path to load pretrainde model from.')
#parser.add_argument('--da_loss', type=str, default='mmd', choices=['mmd','coral'] ,help='Path to load pretrainde model from.')
parser.add_argument('--atten', action='store_true',default=False, help='choose random positives')
parser.add_argument('--relu', type=str, default='relu',choices = ['relu','prelu','sigmoid','softplus','delf_atten'], help='choose random positives')
parser.add_argument('--freeze', action='store_true',default=False, help='choose random positives')
parser.add_argument('--atten_type', type=str,default=None, help='choose random positives')
parser.add_argument('--add_relu', action='store_true',default=False, help='choose random positives')
parser.add_argument('--rect_atten', action='store_true',default=False, help='choose random positives')
parser.add_argument('--mul', type=str,choices=['mul','2mul','3mul','4mul'], help='Path to load pretrainde model from.')
parser.add_argument('--beta', type=float, default='0.002', help='dimention for PCAW')
parser.add_argument('--p_margin', type=float, default='0.5', help='dimention for PCAW')
parser.add_argument('--alpha', type=float, default='0.5')
parser.add_argument('--ins_norm', action='store_true',default=False, help='choose random positives')
parser.add_argument('--debug', action='store_true',default=False)
parser.add_argument('--ratio', type=int, default=4, help='dimention for PCAW')

def train(epoch):
    epoch_loss = 0
    startIter = 1 # keep track of batch iter across subsets for logging

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        #TODO randomise the arange before splitting?
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
        target_subsetIdx = np.array_split(np.arange(len(target_train_set)),subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):
        print('====> Building Cache')
        model.eval()
        train_set.cache = join(opt.cachePath, train_set.whichSet+str(opt.ratio)+str(opt.batchSize)+'ad__'+str(opt.mul)+str(opt.pooling)+str(opt.alpha)+str(opt.p_margin)+str(opt.beta)+str(opt.DA)+str(opt.random_pos_level)+str(opt.freeze)+opt.optim +str(opt.random_crop)+'_'+str(opt.casa)+str(opt.pooling)+str(opt.margin)+str(opt.relu)+str(opt.fromscratch) +'_'+str(opt.num_PCA)+'_'+str(opt.cacheRefreshRate)+'_'+str(opt.lr) +str(opt.arch) +str(opt.atten_type)+'_feat_cache.hdf5')
        #if isfile(train_set.cache):
        #    print('Cache already existed')
        if True:
        #else:
            start_time = time.time()
            with h5py.File(train_set.cache, mode='w') as h5: 
                pool_size = opt.num_PCA
                if pool_size==-1:
                    pool_size = encoder_dim*opt.num_clusters

                h5feat = h5.create_dataset("features", 
                        [len(whole_train_set), pool_size], 
                        dtype=np.float32)
                with torch.no_grad():
                    for iteration, (input, indices) in enumerate(whole_training_data_loader, 1):
                        #print(input.size(),pool_size)
                        input = input.to(device=device,dtype=torch.float)
                        source = Variable(input)
                        vlad_encoding = model(source,source,True)
                        #print(vlad_encoding.size())
                        h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                        del input,source, vlad_encoding
                    #compute pca parameters
                    
            end_time = time.time()-start_time
            print('building cache elasped ', end_time)
        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])
        sub_target_train_set = Subset(dataset=target_train_set, indices = target_subsetIdx[subIter])

        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads, 
                    batch_size=opt.batchSize, shuffle=True, 
                    collate_fn=dataset.collate_fn, pin_memory=cuda)

        target_train_data_loader = DataLoader(dataset=sub_target_train_set, num_workers=opt.threads,
                    batch_size=24, shuffle = True)
        del sub_train_set,sub_target_train_set
        print('*****')
        model.train()
        result=[]
        for iteration, (target,source) in enumerate(zip(target_train_data_loader,training_data_loader),startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            # print(iteration)
            # print(len(target_data),len(source))
            
            target_data = target
            (query, positives, negatives,negCounts, indices) = source
            #print(iteration)
            if query is None: continue # in case we get an empty batch

            B, C, H, W = query.shape

            nNeg = torch.sum(negCounts)

            input = torch.cat([query, positives, negatives])
            

            source_data = input.to(device=device,dtype=torch.float)
            target_data = target_data.to(device= device,dtype=torch.float)
            target_data = Variable(target_data)
            source_data = Variable(source_data)
            del input,query, positives, negatives
            out1, out2 = model(source_data, target_data,False,'feature')

            out2 = out2[:out1.size(0),...]

            mmd_loss = myloss.MK_MMD(Flatten()(out1),Flatten()(out2))
            del out2
            out1 = model(out1,out1,False,'vlad')
            vladQ, vladP, vladN = torch.split(out1, [B, B, nNeg])
            del out1

            optimizer.zero_grad()
            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to 
            # do it per query, per negative
            ranking_loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    ranking_loss += criterion(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1]).cuda()

            ranking_loss /= nNeg.float().to(device) # normalise by actual number of negatives
            if opt.beta > 0.0:
                p_loss = p_criterion(vladQ, vladP).cuda()
                beta = opt.beta
            elif opt.beta <0.0:
                beta = epoch/opt.nEpochs
                p_loss = p_criterion(vladQ, vladP).cuda()
            else:
                p_loss = 0
                beta = 0
            if opt.alpha == 1.0:
                alpha = 0.5*(epoch-1)/opt.nEpochs
            else:
                alpha = opt.alpha
            sum_loss = mmd_loss*alpha + beta*p_loss + (1-beta)*ranking_loss
            sum_loss.backward()
            optimizer.step()

            del  vladQ, vladP, vladN
           # del query, positives, negatives


            batch_loss = sum_loss.item()
            epoch_loss += batch_loss

            result.append({
            'epoch': epoch,
            'step': iteration + 1,
            'total_steps': nBatches,
            #  'lambda': _lambda,
            'mmd_loss': mmd_loss.item(),
            'ranking_loss': ranking_loss.item(),
            #'p_loss': p_loss
             })

            if iteration % 50 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): sum loss: {:.4f} mmd_loss:{:.4f} p_loss:{:.4f} ranking_loss:{:.4f} ".format(epoch, iteration, 
                    nBatches, batch_loss, mmd_loss, p_loss, ranking_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss, 
                        ((epoch-1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg, 
                        ((epoch-1) * nBatches) + iteration)
            del ranking_loss, mmd_loss,sum_loss,batch_loss
            torch.cuda.empty_cache()
        startIter += len(training_data_loader)
        del training_data_loader,target_train_data_loader
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        #remove(train_set.cache) # delete HDF5 cache

    avg_loss = epoch_loss / nBatches
    #avg_mmd_loss = sum(i['mmd_loss'] for i in result)/nBatches
    avg_ranking_loss = sum(i['ranking_loss'] for i in result)/nBatches
    #avg_p_loss = sum(i['p_loss'] for i in result)/nBatches
    print("===> Epoch {} Complete: Avg. Loss: {:.4f} ranking_loss:{:.4f}".format(epoch,
           avg_loss,avg_ranking_loss), 
            flush=True)
    print('beta: {:.4f}, alpha: {:.4f}'.format(beta,alpha),flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def test(eval_set, epoch=0,beelbank=False, write_tboard=False):
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        #pool_size = encoder_dim
        #if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
        pool_size = opt.num_PCA
        if pool_size <0:
            pool_size = opt.num_clusters*encoder_dim
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device=device,dtype=torch.float)
            vlad_encoding = model.forward(input,input,True)
            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(test_data_loader)), flush=True)

            del input, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')
    
    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1,5,10,20,25]

    _, predictions = faiss_index.search(qFeat, max(n_values)) 

    # for each query get those within threshold distance
    if beelbank or opt.split=='beelbank':
        gt = eval_set.getHardPositives()
    else:
        gt = eval_set.getPositives() 

    correct_at_n = np.zeros(len(n_values))
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
    #    if opt.mode in ['test']:

        for i,n in enumerate(n_values):
            if opt.mode in ['test']:
                print(qIx,pred[:n])
                print('----------------')

            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / eval_set.dbStruct.numQ

    recalls = {} #make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard: writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)

    return recalls

def get_clusters(cluster_set):
    nDescriptors = 50000
    nPerImage = 50
    nIm = ceil(nDescriptors/nPerImage)
    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda,
                sampler=sampler)

    if not exists(join(opt.dataPath, 'centroids')):
        makedirs(join(opt.dataPath, 'centroids'))

    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + cluster_set.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", 
                        [nDescriptors, encoder_dim], 
                        dtype=np.float32)
           # print(len(data_loader))
            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(device=device,dtype=torch.float)
                print(input.size())
                image_descriptors = model.base_model(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)
                print(image_descriptors.size())
                print(model.base_model(input).view(input.size(0), encoder_dim, -1).size())
                print(model.base_model(input).size())
                print(image_descriptors.size(1))
                batchix = (iteration-1)*opt.cacheBatchSize*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    #print(encoder_dim)
                    #print(model.encoder.children())
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(nIm/opt.cacheBatchSize)), flush=True)
                del input, image_descriptors
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters, niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

if __name__ == "__main__":
    opt = parser.parse_args()
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
   # print(torch.cuda.get_device_name(1))
   # print(torch.cuda.current_device())

    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 
            'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling', 'optim',
            'margin', 'seed', 'patience']
    if opt.resume:
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {'--'+k : str(v) for k,v in json.load(f).items() if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these 
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del: del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    print(opt)

    if opt.dataset.lower() == 'pittsburgh':
        import pittsburgh as dataset
    elif opt.dataset.lower() == 'mapillary':
        import mapillary_casa_da as dataset
    elif opt.dataset.lower() == 'mnist':
        import mnist as dataset
    elif opt.dataset.lower() == 'm2u':
        import m2u as dataset
    #elif opt.DA:
    #    import m2u as dataset
    else:
        raise Exception('Unknown dataset')

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading dataset(s)')
    if opt.mode.lower() == 'train':
        whole_train_set = dataset.get_whole_training_set(onlyDB=False)

        whole_training_data_loader = DataLoader(dataset=whole_train_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

        train_set = dataset.get_training_query_set(opt.margin)
        if opt.DA:
            if opt.dataset =='m2u':
                target_train_set = dataset.get_usps_dataset('train')
            if opt.dataset == 'mapillary':
                print('getting target train set')
                target_train_set = dataset.get_beelbank_set()



        print('====> Training query set:', len(train_set))
        whole_test_set = dataset.get_whole_val_set()
        whole_bb_test_set = dataset.get_whole_beelbank_test_set()
        print('===> Evaluating on val set, query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'test' :
        if opt.split.lower() == 'test':
            whole_test_set = dataset.get_whole_test_set()
            print('===> Evaluating on beelbank100 set')
        elif opt.split.lower() == 'test250k':
            whole_test_set = dataset.get_250k_test_set()
            print('===> Evaluating on test250k set')
        elif opt.split.lower() == 'train':
            whole_test_set = dataset.get_whole_training_set()
            print('===> Evaluating on train set')
        elif opt.split.lower() == 'val':
            whole_test_set = dataset.get_whole_val_set()
            print('===> Evaluating on val set')
        elif opt.split.lower()=='beelbank':
            whole_test_set = dataset.get_whole_beelbank_test_set()
            print('=====> Evaluating on beelbank test set')

        else:
            raise ValueError('Unknown dataset split: ' + opt.split)
        print('====> Query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'cluster':
        whole_train_set = dataset.get_whole_training_set(onlyDB=True)

    print('===> Building model')

    pretrained = not opt.fromscratch

    if opt.arch.lower() == 'self_define':
        print('using customized network')
        if opt.dataset in ['mapillary','mapillary40k']:
            encoder_dim = 256
            if opt.casa:
                net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=False).cuda()
                print('using Channel attention and spatial attention model')
                model = network.AttentAlex(net_vlad,opt.num_clusters,encoder_dim,opt.num_PCA,opt.bn).cuda()
            elif opt.atten:
                print('using Spatial Attention')
                if opt.rect_atten:
                    print('using rectified attention')
                    #print(opt.pooling)
                    if opt.pooling in ['atten_vlad']:
                    #    print('using attention vlad!')
                        pca_module=None
                        if opt.num_PCA >0:
                            print('using pca module')
                            pca_module = layers.WPCA(opt.num_clusters*encoder_dim,opt.num_PCA)
                            pca_module.init_params(opt.dataset)

                        net_vlad = netvlad.AttenNetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, 
                                         attention=opt.atten_type,mul=opt.mul,actv=opt.relu, vladv2=False,ratio = opt.ratio)
                        model = network.AttenVLADAlex(net_vlad,opt.num_clusters,encoder_dim,opt.relu,
                                 opt.mode,opt.freeze,pretrained,pca_module,opt.ins_norm).cuda()

                    else:
                        net_vlad = netvlad.NetVLAD_res(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=False).cuda()

                        model = network.RecAttentionAlex(net_vlad,opt.num_clusters,encoder_dim,opt.relu,opt.mode,opt.add_relu,
                                                   opt.atten_type,opt.freeze,opt.rect_atten,opt.num_PCA).cuda()
                else:
                    model = network.AttenPoolingAlex(net_vlad,opt.num_clusters,encoder_dim,opt.relu,opt.mode,opt.add_relu,
                                   opt.atten_type,opt.freeze).cuda()
            #elif opt.atten_vlad:
            #    print('using Attention vlad')
            #    net_vlad = netvlad.AttenNetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, attention=opt.attention,vladv2=False).cuda()

            else:
                net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=False).cuda()
                model = network.EmbedNet2(net_vlad,opt.num_clusters,encoder_dim,opt.num_PCA,opt.bn).cuda()
            #if opt.num_PCA>0:
            #    pca_module = network.WPCA(opt.num_clusters*encoder_dim,opt.num_PCA).cuda()
            #    pca_module.init_params(whole_train_set,whole_training_data_loader,model,opt.arch)
            #    model = network.EmbedNet2(net_vlad,opt.num_clusters,encoder_dim,opt.num_PCA,opt.bn,pca_module).cuda()
            #    print('using pca whitening')
            #    print('using embednet2')
#----------------------------------------
#    elif opt.arch.lower() == 'self_define':
#        print('using customized network')

 #       if opt.dataset == 'mapillary':
  #          encoder_dim = 256
  #          net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=False).cuda()
  #          model = network.EmbedNet2(net_vlad,opt.num_clusters,encoder_dim,opt.num_PCA,opt.bn).cuda()
  #          print('using embednet2')
  #      else:
  #          print('using mynet')
  #          encoder_dim = 64
  #          net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=False).cuda()
  #          model = network.MyNet(net_vlad,opt.num_clusters,encoder_dim,opt.num_PCA,opt.bn).cuda()
        #layers = list(model.base_model.children())

    if opt.mode.lower() == 'cluster': #and opt.vladv2 == False #TODO add v1 v2 switching as flag
        if opt.dataset == 'mapillary':
            model = network.EmbedNet2(net_vlad,opt.num_clusters,encoder_dim,opt.num_PCA,opt.bn).cuda()
        else:
            model = network.MyNet(net_vlad,opt.num_clusters,encoder_dim,opt.num_PCA,opt.bn).cuda()
    if opt.pretrain:
        if opt.ckpt.lower() == 'latest':
            pretrain_ckpt = join(opt.pretrain, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            pretrain_ckpt = join(opt.pretrain, 'checkpoints', 'model_best.pth.tar')
            

        if isfile(pretrain_ckpt):
            print("=> loading checkpoint '{}'".format(pretrain_ckpt))
            checkpoint = torch.load(pretrain_ckpt, map_location=lambda storage, loc: storage)

            if opt.nGPU > 1 and torch.cuda.device_count() > 1:
                new_state_dict = OrderedDict()
                for k,v in checkpoint['state_dict'].items():
                    if k[:7]=='encoder':
                        name = k[:7]+'.module'+k[7:] # remove `module.`
                    elif k[:4] == 'pool':
                        name = k[:4]+'.module'+k[4:]
                    #print name
                    new_state_dict[name] = v
            else: 
                new_state_dict = checkpoint['state_dict']
            model.load_state_dict(new_state_dict)
          #  model = model.to(device)

          #  if opt.mode == 'train':
          #      optimizer.load_state_dict(checkpoint['optimizer'])
          #      if opt.nGPU > 1 and torch.cuda.device_count() > 1:
          #          for state in optimizer.state.values():
           #             for k, v in state.items():
           #                 if torch.is_tensor(v):
           #                     state[k] = v.cuda()
            print("=> loaded pretrainde model'{}' (start epoch {})"
                  .format(pretrain_ckpt, opt.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(pretrain_ckpt))

    if opt.DA:
        model = network.DeepCORAL2(model).cuda()

    if opt.mode.lower() != 'cluster':
        #if opt.DA:
        #    model = network.DeepCORAL(model).cuda()
        if opt.pooling.lower() in ['netvlad','atten_vlad']:
            net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=True).cuda()
            if not opt.resume: 
                if opt.mode.lower() == 'train':
                    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + train_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')
                else:
                    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + whole_test_set.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')

                if not exists(initcache):
                    raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')

                with h5py.File(initcache, mode='r') as h5: 
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    net_vlad.init_params(clsts, traindescs) 
                    del clsts, traindescs

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids=[0,1])
        #if opt.mode.lower() != 'cluster':
         #   model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)
    
    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr)#, betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        # original paper/code doesn't sqrt() the distances, we do, so sqrt() the margin, I think :D
        #criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, 
        #        p=2, reduction='sum').to(device)
        if opt.loss == 'l2':
            criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, 
                p=2, reduction='sum').to(device)
        elif opt.loss == 'cos':
            print('using cosine similarity as metric')
            criterion = myloss.triplet_cos_loss(margin=opt.margin).to(device)
        elif opt.loss =='impr_triplet':
            print('using improved triplet loss')
            criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, 
                p=2, reduction='sum').to(device)
            p_criterion = myloss.positive_loss(opt.p_margin).to(device)

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            if opt.mode == 'train':
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))


    #if opt.DA:
    #        model = network.DeepCORAL2(model).cuda()

    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = 1
        recalls = test(whole_test_set, epoch, write_tboard=False)
    elif opt.mode.lower() == 'cluster':
        print('===> Calculating descriptors and clusters')
        get_clusters(whole_train_set)
    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+str(opt.alpha)))

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        #opt.savePath = join(logdir, opt.savePath)
        
        if opt.resume:
            opt.savePath = opt.savePath
        else:
            opt.savePath = join(logdir, opt.savePath)

        if not opt.resume:
            makedirs(opt.savePath)

        with open(join(opt.savePath, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)

        not_improved = 0
        best_score = 0
        for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            train(epoch)
            if (epoch % opt.evalEvery) == 0 or epoch in [1,3]:
                recalls = test(whole_test_set, epoch, write_tboard=True)
                is_best = recalls[5] > best_score 
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else: 
                    not_improved += 1

                save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'recalls': recalls,
                        'best_score': best_score,
                        'optimizer' : optimizer.state_dict(),
                        'parallel' : isParallel,
                }, is_best)

                recalls1 = test(whole_bb_test_set, epoch,True, write_tboard=True)


        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()


import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image
import os
from sklearn.neighbors import NearestNeighbors
import h5py
import faiss
root_dir = '../data/mapillary/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')

struct_dir = join(root_dir, 'datasets/')
queries_dir = root_dir
img_dir = root_dir
target_path = '../data/beelbank_ccorp/'

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])
def random_crop():
    return transforms.Compose([
        transforms.RandomResizedCrop(521, scale=(0.3,1.0),ratio=(1.0, 1.0), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),

        ])
def get_whole_training_set(onlyDB=False):
    structFile = join(struct_dir, 'dbMapillary_20k_train.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform(),
                             onlyDB=onlyDB)
def get_whole_val_set():
    structFile = join(struct_dir, 'dbMapillary_20k_val.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())
def get_whole_test_set():
    structFile = join(struct_dir, 'dbMapillary_20k_test.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())
def get_whole_beelbank_test_set():
    structFile = join(struct_dir, 'beelbank1k_test.mat')
    return WholeDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_training_query_set(margin=0.1,random_level=5,Random_crop=False):
    structFile = join(struct_dir, 'dbMapillary_20k_train.mat')
    return QueryDatasetFromStruct(structFile,random_level=random_level,Random_crop=Random_crop,
                             input_transform=input_transform(), margin=margin)

def get_val_query_set():
    structFile = join(struct_dir, 'dbMapillary_20k_val.mat')
    return QueryDatasetFromStruct(structFile,
                             input_transform=input_transform())

def get_beelbank_set():
    
    return MyDataset()

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 
    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])


def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct']

    dataset = 'mapillary'
    
    whichSet = matStruct['whichSet'].item().item()
    
    dbImage = [f[0].item() for f in matStruct['dbImageFns'].item()]
    utmDb = matStruct['utmDb'].item().T

    qImage = [f[0].item() for f in matStruct['qImageFns'].item()]
    utmQ = matStruct['utmQ'].item().T

    numDb = matStruct['numImage'].item()[0][0]
    numQ = matStruct['numQueries'].item()[0][0]

    #posDistThr = matStruct['posDistThr'].item()[0][0]
    #posDistSqThr = matStruct['posDistSqThr'].item()[0][0]
    #nonTrivPosDistSqThr = matStruct['nonTrivPosDistSqThr'].item()[0][0]
    posDistThr = 25   #select positives and potential negative
    posDistSqThr = 625
    nonTrivPosDistSqThr = 100

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, 
            utmQ, numDb, numQ, posDistThr, 
            posDistSqThr, nonTrivPosDistSqThr)

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, structFile, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct(structFile)
        self.images = [join(root_dir,dbIm.split('/')[1],dbIm.split('/')[2]) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(queries_dir, qIm.split('/')[1],qIm.split('/')[2]) for qIm in self.dbStruct.qImage]

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None
        self.cache = None

    def __getitem__(self, index):
        #path = join(root_dir,self.images[index])
        #with Image.open(self.images[index]) as I:
        I = Image.open(self.images[index],mode='r')
            #if self.input_transform:
        img = self.input_transform(I)
        I.close()
        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=1)
            knn.fit(self.dbStruct.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=self.dbStruct.posDistThr)

        return self.positives
    def getHardPositives(self):
        if self.positives is None:
            self.positives = np.load('positives.npy',encoding="latin1")
        return self.positives
        
def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x:x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices

class QueryDatasetFromStruct(data.Dataset):
    def __init__(self, structFile ,random_level,Random_crop, nNegSample=1000, nNeg=10, margin=0.1, input_transform=None):
        super().__init__()
        if Random_crop:
            self.input_transform = random_crop()
            print('random crop the input images')
        else:
            self.input_transform = input_transform
        self.margin = margin

        self.dbStruct = parse_dbStruct(structFile)
        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset
        self.nNegSample = nNegSample # number of negatives to randomly sample
        self.nNeg = nNeg # number of negatives used for training
        self.random_level=random_level

        # potential positives are those within nontrivial threshold range
        #fit NN to find them, search by radius
        knn = NearestNeighbors(n_jobs=1)
        knn.fit(self.dbStruct.utmDb)

        # TODO use sqeuclidean as metric?
        
        self.nontrivial_positives = list(knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.nonTrivPosDistSqThr**0.5, 
                return_distance=False))
        # radius returns unsorted, sort once now so we dont have to later
        for i,posi in enumerate(self.nontrivial_positives):
            self.nontrivial_positives[i] = np.sort(posi)
        # its possible some queries don't have any non trivial potential positives
        # lets filter those out
        self.queries = np.where(np.array([len(x) for x in self.nontrivial_positives])>0)[0]

        # potential negatives are those outside of posDistThr range
        potential_positives = knn.radius_neighbors(self.dbStruct.utmQ,
                radius=self.dbStruct.posDistThr, 
                return_distance=False)

        self.potential_negatives = []
        for pos in potential_positives:
            self.potential_negatives.append(np.setdiff1d(np.arange(self.dbStruct.numDb),
                pos, assume_unique=True))

        self.cache = None # filepath of HDF5 containing feature vectors for images

        self.negCache = [np.empty((0,)) for _ in range(self.dbStruct.numQ)]

    def __getitem__(self, index):
        index = self.queries[index] # re-map index to match dataset
        #start_time = time.time()
        with h5py.File(self.cache, mode='r') as h5: 
            h5feat = h5.get("features")

            qOffset = self.dbStruct.numDb
            qFeat = h5feat[index+qOffset]

            posFeat = h5feat[self.nontrivial_positives[index].tolist()]
            d = qFeat.shape[-1]
            faiss_index = faiss.IndexFlatL2(d)
            faiss_index.train(posFeat)
            faiss_index.add(posFeat)
            dPos, posNN = faiss_index.search(qFeat.reshape(1,-1), self.random_level) 
            #dPos, posNN = knn.kneighbors(qFeat.reshape(1,-1), 1)
            select = np.random.choice(self.random_level,1)
            #dPos = dPos.item()
            dPos = dPos[0][select]
            posIndex = self.nontrivial_positives[index][posNN[0][select]].item() #random choose a positive sample

            negSample = np.random.choice(self.potential_negatives[index], self.nNegSample)
            negSample = np.unique(np.concatenate([self.negCache[index], negSample]))

            negFeat = h5feat[negSample.tolist()]

            faiss_index1 = faiss.IndexFlatL2(d)
            faiss_index1.train(negFeat)
            faiss_index1.add(negFeat)
            dNeg, negNN = faiss_index1.search(qFeat.reshape(1,-1), self.nNeg)
            #dNeg, negNN = knn.kneighbors(qFeat.reshape(1,-1), 
             #       self.nNeg*10) # to quote netvlad paper code: 10x is hacky but fine
            dNeg = dNeg.reshape(-1)
            negNN = negNN.reshape(-1)

            # try to find negatives that are within margin, if there aren't any return none
            violatingNeg = dNeg < dPos + self.margin**0.5
     
            if np.sum(violatingNeg) < 1:
                #if none are violating then skip this query
                return None

            negNN = negNN[violatingNeg][:self.nNeg]
            negIndices = negSample[negNN].astype(np.int32)
            self.negCache[index] = negIndices
        query_path = join(queries_dir, self.dbStruct.qImage[index].split('/')[1],self.dbStruct.qImage[index].split('/')[2])
        positive_path = join(root_dir, self.dbStruct.dbImage[posIndex].split('/')[1],self.dbStruct.dbImage[posIndex].split('/')[2])
        I = Image.open(query_path,mode='r')
        query = self.input_transform(I)
        I.close()
        I = Image.open(positive_path,mode='r')
        positive = self.input_transform(I)

        I.close()
       # with Image.open(query_path) as I:
       #     query = self.input_transform(I)
       # with Image.open(positive_path) as I:
       #     positive = self.input_transform(I)
        #query = Image.open(join(queries_dir, self.dbStruct.qImage[index].split('/')[1],self.dbStruct.qImage[index].split('/')[2]))
        #positive = Image.open(join(root_dir, self.dbStruct.dbImage[posIndex].split('/')[1],self.dbStruct.dbImage[posIndex].split('/')[2]))

        #if self.input_transform:
        #    query = self.input_transform(query)
        #    positive = self.input_transform(positive)

        negatives = []
        for negIndex in negIndices:
            negative_path = join(root_dir, self.dbStruct.dbImage[negIndex].split('/')[1], self.dbStruct.dbImage[negIndex].split('/')[2])
            #negative = Image.open(join(root_dir, self.dbStruct.dbImage[negIndex].split('/')[1], self.dbStruct.dbImage[negIndex].split('/')[2]))
            I = Image.open(negative_path,mode='r')
           # with Image.open(negative_path) as I:
            #if self.input_transform:
            negative = self.input_transform(I)
            I.close()
            negatives.append(negative)

        negatives = torch.stack(negatives, 0)
        return query, positive, negatives, [index, posIndex]+negIndices.tolist()

    def __len__(self):
        return len(self.queries)

class MyDataset(data.Dataset):
    def __init__(self, dir_=target_path,input_transform=input_transform()):
        super().__init__()
        self.transform = input_transform
        self.dir_ = dir_
        img_list = os.listdir(dir_)
        self.images = img_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        I = Image.open(self.dir_+self.images[index],mode='r')
        #with Image.open(self.dir_+self.images[index]) as I:
        #X = Image.open(self.dir_+self.images[index])
        X = self.transform(I)
        I.close()
        return X

def GetTargetData(target_path,batch_size):
    input_transform = transforms.Compose([
        #transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])
    dataset = MyDataset(dir_=target_path,input_transform=input_transform)
    data_loader = data.DataLoader(dataset,batch_size=batch_size,num_workers=2,shuffle=True)


    return data_loader

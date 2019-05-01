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

from tensorboardX import SummaryWriter
import numpy as np
import netvlad
import GB

parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster'])
parser.add_argument('--batchSize', type=int, default=1,
        help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=1, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000, 
        help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
        help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=1, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='data/', help='Path for centroid data.')
parser.add_argument('--runsPath', type=str, default='runs/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints', 
        help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default='/tmp', help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest', 
        help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=1, 
        help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
parser.add_argument('--dataset', type=str, default='GB',
        help='Dataset to use', choices=['GB'])
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet'])
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use', 
        choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val', 
        choices=['test', 'test250k', 'train', 'val'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')



def test(eval_set, qFeat, idxDb, write_tboard=False):
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        name = join('/data/SenseTimeLocalizationData_GB/DatabaseNetVLAD', GB.dbFolders[idxDb] + '_db_feat.hdf5')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))
        if exists(name):
            with h5py.File(name, mode='r') as h5:
                #print(list(h5.keys()))
                h5feat = h5['features']
                #print(h5feat.shape)
                dbFeat = h5feat[:eval_set.numDb]

        else:
            print('====> Extracting Features')
            with h5py.File(name, mode='w') as h5:
                h5feat = h5.create_dataset("features",
                                           [len(eval_set), pool_size],
                                           dtype=np.float32)
                for iteration, (input, indices) in enumerate(test_data_loader, 1):
                    input = input.to(device)
                    image_encoding = model.encoder(input)
                    vlad_encoding = model.pool(image_encoding)

                    h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    if iteration % 50 == 0 or len(test_data_loader) <= 10:
                        print("==> Batch ({}/{})".format(iteration,
                            len(test_data_loader)), flush=True)

                    del input, image_encoding, vlad_encoding
                    if iteration == eval_set.numDb//test_data_loader.batch_size+1:
                        break
                dbFeat = h5feat[:eval_set.numDb]
    del test_data_loader


    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    distances, predictions = faiss_index.search(qFeat, 1)

    return distances, predictions

def getQueryFeatures(eval_set, epoch=0, write_tboard=False):
    # TODO what if features dont fit in memory?
    test_data_loader = DataLoader(dataset=eval_set,
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
        dbFeat = np.empty((eval_set.numQ, pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding)

            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration,
                    len(test_data_loader)), flush=True)

            del input, image_encoding, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat.astype('float32')
    print('--Got qFeat--')

    return qFeat


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
    elif opt.dataset.lower() == 'gb':
        import GB as dataset
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
    if opt.mode.lower() == 'test':
        whole_test_set = dataset.get_query_set()
        print('====> Query count:', whole_test_set.numQ)
    else:
        raise Exception('Unknown mode')

    print('===> Building model')
    pretrained = not opt.fromscratch
    if opt.arch.lower() == 'alexnet':
        encoder_dim = 256
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'vgg16':
        encoder_dim = 512
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]
        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False

    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)

    if opt.pooling.lower() == 'netvlad':
        net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=False)
        model.add_module('pool', net_vlad)
    elif opt.pooling.lower() == 'max':
        global_pool = nn.AdaptiveMaxPool2d((1, 1))
        model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
    elif opt.pooling.lower() == 'avg':
        global_pool = nn.AdaptiveAvgPool2d((1, 1))
        model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
    else:
        raise ValueError('Unknown pooling type: ' + opt.pooling)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)


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
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = 1
        distances = np.empty((whole_test_set.numQ, len(dataset.dbFolders)))
        predictions = np.empty((whole_test_set.numQ, len(dataset.dbFolders)))
        import time
        start = time.time()
        qFeat = getQueryFeatures(whole_test_set, epoch, write_tboard=False)
        end = time.time()
        print(end-start)
        for idxDb in range(0, len(dataset.dbFolders)):
            distance, prediction = test(dataset.get_sub_set(idxDb), qFeat, idxDb, write_tboard=False)
            distances[:, idxDb] = distance[:,0]
            predictions[:, idxDb] = prediction[:,0]
        np.savetxt('distances_3F_bright.txt', distances)
        np.savetxt('predictions_3F_bright.txt', predictions)

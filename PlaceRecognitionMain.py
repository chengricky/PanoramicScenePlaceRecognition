"""
Test place recognition network (NetVLAD), which is applicable to unwrapped panoramic images
"""

from __future__ import print_function
import argparse
import random
from os.path import join, isfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import faiss
import numpy as np
import netvlad


parser = argparse.ArgumentParser(description='ScenePlaceRecognitionTest')
parser.add_argument('--cacheBatchSize', type=int, default=8, help='Batch size for caching and testing')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='data/', help='Path for centroid data.')
parser.add_argument('--cachePath', type=str, default='/tmp/', help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='checkpoints_res',
                    help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='best',
                    help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--dataset', type=str, default='Yuquan',
                    help='DataSet to use', choices=['MOLP', 'Yuquan', 'Highway'])
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
                    choices=['netvlad', 'max', 'avg'])
parser.add_argument('--fusion', type=str, default='concat', help='how to fuse multiple descriptors',
                    choices=['add', 'concat'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--netVLADtrainNum', type=int, default=2, help='Number of trained blocks in Resnet18.')
parser.add_argument('--panoramicCrop', type=int, default=5, help='Number of panoramic crops')


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


def get_input_batches(input):
    """Convert panoromas into split images for CNN input"""
    inN, inC, inH, inW = input.size()
    sub_width = inW // opt.panoramicCrop
    input_batches = torch.empty(opt.panoramicCrop * inN, inC, inH, sub_width)
    input_batches.requires_grad = True
    for bt in range(inN):
        for idx in range(opt.panoramicCrop):
            input_batches[bt * opt.panoramicCrop + idx, :, :, :] \
                = input[bt, :, :, idx * sub_width:(idx + 1) * sub_width]
    return input_batches


def generate_vlad(batches, inN):
    """Merge the descriptors of split images into panoramic NetVLAD

    :param batches: the network output of split images
    :param inN: the batch size (number of panorama)
    :return: panoramic NetVLAD
    """
    if opt.pooling.lower() == 'netvlad' and opt.fusion.lower() == 'add':
        vlad_encoding = torch.zeros(inN, batches.shape[1], batches.shape[2])
        vlad_encoding.requires_grad = True
        for i in range(inN):
            for j in range(opt.panoramicCrop):
                vlad_encoding[i, :, :] += batches[i * opt.panoramicCrop + j, :, :].detach().cpu()
        vlad_encoding = F.normalize(vlad_encoding, p=2, dim=2)  # intra-normalization
        vlad_encoding = vlad_encoding.view(inN, -1)  # flatten
        vlad_encoding = F.normalize(vlad_encoding, p=2, dim=1)  # L2 normalize
    else:
        if opt.pooling.lower() == 'netvlad':
            batches = F.normalize(batches, p=2, dim=2)  # intra-normalization
            batches = batches.view(batches.shape[0], -1)  # flatten
            batches = F.normalize(batches, p=2, dim=1)  # L2 normalize
        vlad_encoding = torch.zeros(inN, batches.shape[1] * opt.panoramicCrop)
        print(batches.shape[0])
        for i in range(inN):
            for j in range(opt.panoramicCrop):
                vlad_encoding[i,
                batches.shape[1] * j:batches.shape[1] * (j + 1)] = \
                    batches[i * opt.panoramicCrop + j, :].detach().cpu()
        vlad_encoding = F.normalize(vlad_encoding, p=2, dim=1)  # L2 normalize
    return vlad_encoding


def test_dataset(eval_set, output_feats=False):
    test_data_loader = DataLoader(dataset=eval_set, num_workers=opt.threads,
                                  batch_size=opt.cacheBatchSize, shuffle=False, pin_memory=cuda)
    model.eval()
    # 不会反向传播，提高inference速度
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad' and opt.fusion.lower() == 'add':
            pool_size *= opt.num_clusters
        elif opt.pooling.lower() == 'netvlad':
            pool_size *= opt.num_clusters*opt.panoramicCrop
        else:
            pool_size *= opt.panoramicCrop
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            inN, inC, inH, inW = input.size()
            sub_width = inW // opt.panoramicCrop
            input_batches = torch.empty(opt.panoramicCrop*inN, inC, inH, sub_width)
            for bt in range(inN):
                for idx in range(opt.panoramicCrop):
                    input_batches[bt*opt.panoramicCrop+idx, :, :, :] \
                        = input[bt, :, :, idx*sub_width:(idx+1)*sub_width]
            del input
            input_batches = input_batches.to(device)

            # forward pass - netvlad
            image_encoding = model.encoder(input_batches)
            vlad_encoding_batches = model.pool(image_encoding)

            vlad_encoding = generate_vlad(vlad_encoding_batches, inN)

            dbFeat[indices.detach().numpy(), :] = vlad_encoding.numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, len(test_data_loader)), flush=True)

            del input_batches, vlad_encoding, image_encoding, vlad_encoding_batches
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.numDb].astype('float32')

    if output_feats:
        np.savetxt("query.txt", qFeat)
        np.savetxt("database.txt", dbFeat)

    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20]
    distances, predictions = faiss_index.search(qFeat, max(n_values))

    # for each query get those within threshold distance
    gt = eval_set.getPositives()

    correct_at_n = np.zeros(len(n_values))
    numP = 0
    # TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        if not gt[qIx]:
            continue
        else:
            numP += 1
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / numP

    recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))


# loading ResNet18 of trained on places365 as the backbone
def baseResNet(type=50):
    from Place365 import wideresnet
    if type == 18:
        model_res_net = wideresnet.resnet18(num_classes=365)
    elif type == 34:
        model_res_net = wideresnet.resnet34()
    elif type == 50:
        model_res_net = wideresnet.resnet50(num_classes=365)
    else:
        raise Exception('Unknown ResNet Type')
    layers = list(model_res_net.children())[:-2]  # children()只包括了第一代儿子模块，get rid of the last two layers: avepool & fc
    return layers


if __name__ == "__main__":
    opt = parser.parse_args()

    # designate device
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    # designate dataset
    if opt.dataset.lower() == 'molp':
        from DataSet import MOLP as dataset
    elif opt.dataset.lower() == 'yuquan':
        from DataSet import Yuquan as dataset
    elif opt.dataset.lower() == 'highway':
        from DataSet import Highway as dataset
    else:
        raise Exception('Unknown dataset')
    print('===> Loading dataset(s)')
    whole_test_set = dataset.get_whole_val_set(opt.panoramicCrop)
    print('====> Query count:', whole_test_set.numQ)

    # build network architecture: ResNet-18 with scene classification / scene attribute
    print('===> Building model')
    layers = baseResNet(type=18)
    encoder = nn.Sequential(*layers)    # 参数数目不定时，使用*号作为可变参数列表，就可以在方法内对参数进行调用。
    model = nn.Module()
    model.add_module('encoder', encoder)

    # 添加（初始化）pooling模块
    encoder_dim = 512
    if opt.pooling.lower() == 'netvlad':
        net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim,
                                   vladv2=False, normalize_output=False)
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
        model.pool = nn.DataParallel(model.pool)
        isParallel = True

    # load the paramters of the DataSet branch
    if opt.ckpt.lower() == 'latest':
        resume_ckpt = join(opt.resume, 'checkpoint.pth.tar')
    elif opt.ckpt.lower() == 'best':
        resume_ckpt = join(opt.resume, 'model_best.pth.tar')

    if isfile(resume_ckpt):
        print("=> loading checkpoint '{}'".format(resume_ckpt))
        checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']
        state_dict_encoder = {k: v for k, v in state_dict.items() if 'encoder' in k}
        state_dict_encoder = {str.replace(k, 'encoder.', ''): v for k, v in state_dict_encoder.items()}
        model.encoder.load_state_dict(state_dict_encoder, strict=True)
        if opt.pooling.lower() == 'netvlad':
            state_dict_pool = {k: v for k, v in state_dict.items() if 'pool' in k}
            state_dict_pool = {str.replace(k, 'pool.', ''): v for k, v in state_dict_pool.items()}
            model.pool.load_state_dict(state_dict_pool, strict=True)
    else:
        print("=> no checkpoint found at '{}'".format(resume_ckpt))

    model = model.to(device)

    # test
    test_dataset(whole_test_set)



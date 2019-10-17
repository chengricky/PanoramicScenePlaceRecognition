# Test scene place recognition network (NetVLAD), which is applicable to unwrapped panoramic images
from __future__ import print_function
import argparse
import random
from os.path import join, exists

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import h5py
import faiss
import numpy as np
import netvlad
import cv2

import SceneModel
import warnings

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
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--attention', action='store_true', help='Whether with the attention module.')
parser.add_argument('--netVLADtrainNum', type=int, default=2, help='Number of trained blocks in Resnet18.')
parser.add_argument('--panoramicCrop', type=int, default=4, help='Number of panoramic crops')


def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'Place365/categories_places365.txt'
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'Place365/IO_places365.txt'
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'Place365/labels_sunattribute.txt'
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'Place365/W_sceneattribute_wideresnet18.npy'
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute


def returnCAM(feature_conv_b, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    # size_upsample = (256, 256)
    batch, nc, h, w = feature_conv_b.shape
    output_cam = torch.zeros(opt.panoramicCrop, 1, h, w)
    for b in range(0, batch):
        feature_conv = feature_conv_b[b, :, :, :].squeeze()
        for idx in class_idx[b]:
            cam = weight_softmax[idx, :].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(1, h, w)
            cam_img = torch.from_numpy(cam)
            output_cam[b, :, :, :] += cam_img
        output_cam[b, :, :, :] = output_cam[b, :, :, :] - torch.min(output_cam[b, :, :, :])
        output_cam[b, :, :, :] = output_cam[b, :, :, :] / torch.max(output_cam[b, :, :, :])
    return output_cam


def returnAttention(feature_conv_b, weight_softmax, class_idx):
    # get the attention map from places branch
    CAM = returnCAM(SceneModel.features_blobs[0], weight_softmax, class_idx).to(device)
    batch, nc, h, w = feature_conv_b.shape
    output_cam = feature_conv_b.mul(CAM)
    return output_cam


def test_dataset(eval_set, output_feats=False):
    # TODO what if features dont fit in memory?
    test_data_loader = DataLoader(dataset=eval_set,
                                  num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False,
                                  pin_memory=cuda)
    # 不会反向传播，提高inference速度
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
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

            # forward pass - add scene information
            logit = modelPlaces.forward(input_batches)
            h_x = F.softmax(logit, 1).data
            probs, idx = h_x.sort(1, True)
            #probs = probs.cpu().numpy()
            idx = idx.cpu().numpy()
            #CAMs = returnCAM(SceneModel.features_blobs[0], weight_softmax, [idx[0]])  # layer4
            #SceneModel.features_blobs.clear()

            # forward pass - netvlad
            image_encoding = model.encoder(SceneModel.netVLADlayer_input[0])
            SceneModel.netVLADlayer_input.clear()
            # 引入attention机制
            if opt.attention:
                image_encoding = returnAttention(image_encoding, weight_softmax, idx[:, :1])  # layer4
            SceneModel.features_blobs.clear()
            vlad_encoding_batches = model.pool(image_encoding)

            # 将各个batch相加
            vlad_encoding = np.zeros((inN, vlad_encoding_batches.shape[1]))
            for i in range(inN):
                for j in range(opt.panoramicCrop):
                    vlad_encoding[i, :] += vlad_encoding_batches[i*opt.panoramicCrop+j, :].detach().cpu().numpy()

            dbFeat[indices.detach().numpy(), :] = vlad_encoding
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, len(test_data_loader)), flush=True)

            del input_batches, logit, vlad_encoding, image_encoding, vlad_encoding_batches
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
    n_values = [1, 5, 10]
    distances, predictions = faiss_index.search(qFeat, max(n_values))

    # # for each query get those within threshold distance
    # gt = eval_set.getPositives()
    #
    # correct_at_n = np.zeros(len(n_values))
    # # TODO can we do this on the matrix in one go?
    # for qIx, pred in enumerate(predictions):
    #     for i, n in enumerate(n_values):
    #         # if in top N then also in top NN, where NN > N
    #         if np.any(np.in1d(pred[:n], gt[qIx])):
    #             correct_at_n[i:] += 1
    #             break
    # recall_at_n = correct_at_n / eval_set.numQ
    #
    # recalls = {}  # make dict for output
    # for i, n in enumerate(n_values):
    #     recalls[n] = recall_at_n[i]
    #     print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))

    fp = 0
    tp = 0
    for i in range(0, len(predictions)):
        if any(abs(predictions[i] - i) <= 5):
            tp = tp + 1
        else:
            fp = fp + 1

    precision = tp/(tp+fp)
    recall = 1
    f1 = 2*precision/(precision+recall)

    print('F1=', f1)

    return distances, predictions
    # return recalls, distances, predictions


def test(logit, ifPrint=False):

    # output the IO prediction
    io_image = np.mean(labels_IO[idx[:10]])  # vote for the indoor or outdoor
    if ifPrint:
        if io_image < 0.5:
            print('--TYPE OF ENVIRONMENT: indoor')
        else:
            print('--TYPE OF ENVIRONMENT: outdoor')

        # output the prediction of scene category
        print('--SCENE CATEGORIES:')
        for i in range(0, 5):
            print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # output the scene attributes
    responses_attribute = W_attribute.dot(SceneModel.features_blobs[1])
    idx_a = np.argsort(responses_attribute)
    if ifPrint:
        print('--SCENE ATTRIBUTES:')
        print(', '.join([labels_attribute[idx_a[i]] for i in range(-1, -10, -1)]))

        # generate class activation mapping
        print('Class activation map is saved as cam.jpg')


    if ifPrint:
        # render the CAM and output
        img = cv2.imread('test.jpg')
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.4 + img * 0.5
        cv2.imwrite('cam.jpg', result)


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
    modelPlaces = SceneModel.loadSceneRecognitionModel(opt.netVLADtrainNum)
    # build network architecture: ResNet-18 with place recognition
    model = SceneModel.loadPlaceRecognitionEncoder(opt.netVLADtrainNum)
    # 添加（初始化）pooling模块
    encoder_dim = 512
    if opt.pooling.lower() == 'netvlad':
        net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=False)
        initcache = join(opt.dataPath, 'centroids', 'resnet18_pitts30k_' + str(opt.num_clusters) + '_desc_cen.hdf5')
        if not exists(initcache):
            raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')
        with h5py.File(initcache, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            net_vlad.init_params(clsts, traindescs)
            del clsts, traindescs
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
    model = SceneModel.loadNetVLADParams(resume_ckpt, opt.netVLADtrainNum, model)

    # execute test procedures
    print('===> Running evaluation step')
    # scene recognition - load the labels
    classes, labels_IO, labels_attribute, W_attribute = load_labels()
    # scene recognition - get the softmax weight
    params = list(modelPlaces.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax < 0] = 0

    modelPlaces = modelPlaces.to(device)
    model = model.to(device)
    # test
    distances, predictions = test_dataset(whole_test_set)


# read and save the json file of training hyper-parameters
import os
import json
import argparse

parser = argparse.ArgumentParser(description='PlaceRecognitionTrainParameters')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'cluster', 'test'])
parser.add_argument('--saveDecs',  action='store_true', help='whether to save descriptors to files.')
parser.add_argument('--batchSize', type=int, default=3,
                    help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
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
parser.add_argument('--threads', type=int, default=4, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='data/', help='Path for centroeid data.')
parser.add_argument('--runsPath', type=str, default='runs/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints/',
                    help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default='/tmp/', help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='',
                    help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest',
                    help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=1,
                    help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping. 0 is off.')
parser.add_argument('--dataset', type=str, default='pittsburgh',
                    help='Dataset to use', choices=['pittsburgh', 'tokyo247', 'highway', 'GB', 'multimodal'])
parser.add_argument('--arch', type=str, default='resnet18',
                    help='basenetwork to use', choices=['vgg16', 'alexnet', 'resnet18', 'resnet34', 'resnet50'])
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
                    choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--split', type=str, default='test', help='Data split to use for testing. Default is val',
                    choices=['test', 'test250k', 'train', 'val'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
parser.add_argument('--numTrain', type=int, default=2, help='the number of trained layers of basenet')
parser.add_argument('--withAttention', action='store_true', help='Whether with the attention module.')
parser.add_argument('--remain', type=float, default=0.5, help='The remained ratio of feature map.')
parser.add_argument('--vladv2', action='store_true', help='whether to use VLADv2.')
parser.add_argument('--reduction', action='store_true', help='whether to perform PCA dimension reduction.')
parser.add_argument('--panoramicCrop', type=int, default=4, help='Number of panoramic crops')
parser.add_argument('--fusion', type=str, default='add', help='how to fuse multiple descriptors',
                    choices=['add', 'concat'])

def get_args():
    # read arguments from command or json file
    opt = parser.parse_args()
    restore_var = [ 'nGPU', 'arch', 'num_clusters', 'pooling',
                   'margin', 'seed', 'patience', 'vladv2']
    #savePath, lr-ralated ( 'lr', 'lrStep', 'lrGamma')'weightDecay', 'momentum', 'runsPath','optim',
    if opt.resume:
        opt_loaded = read_arguments(opt, parser, restore_var)
        return opt_loaded
    else:
        return opt


def read_arguments(opt, parser_, restore_var):
    flag_file = os.path.join(opt.resume, 'flags.json')
    if os.path.exists(flag_file):
        with open(flag_file, 'r') as f:
            stored_flags = {'--' + k: str(v) for k, v in json.load(f).items() if k in restore_var}

            to_del = []
            for flag, val in stored_flags.items():
                for act in parser_._actions:
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
            opt_load = parser_.parse_args(train_flags, namespace=opt)
    return opt_load

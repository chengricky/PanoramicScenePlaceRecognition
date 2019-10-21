import torchvision.transforms as transforms
import torch.utils.data as data
import os
import faiss
import numpy as np
from PIL import Image
from geopy import distance
# https://pypi.org/project/geopy/

root_dir = '/media/ricky/Entertainment/YuquanPAL4Localization/ProcessedDataset/'
if not os.path.exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Yuquan dataset')

# the list of database folder (images)
dbFolder = root_dir + 'Afternoon1-GPS'
qFolder = root_dir + 'Afternoon2-GPS'

queryGPS = os.path.join(qFolder, "GPS.txt")
databaseGPS = os.path.join(dbFolder, "GPSNew.txt")
databaseOverlap = os.path.join(qFolder, "Overlap.txt")


def read_gnss(path):
    coordinate = []
    with open(path) as file_in:
        for line in file_in:
            strs = line.split()
            lon = float(strs[0])
            lat = float(strs[1])
            coordinate.append([lat, lon])
    return coordinate


def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# 返回单位为m
def gnss_distance(c1, c2):
    return distance.distance(c1, c2).meters


# DB + query
def get_whole_val_set(panoramicCrop):
    return DatasetFromStruct(dbFolder, qFolder, input_transform=input_transform())


class DatasetFromStruct(data.Dataset):
    def __init__(self, dbFolder, qFolder, input_transform=None):
        super().__init__()
# DATABASE 图像
        self.input_transform = input_transform
        list_img = os.listdir(dbFolder)
        list_img = [f for f in list_img if '.jpg' in f]
        self.images = [os.path.join(dbFolder, dbIm) for dbIm in list_img]
        self.numDb = len(self.images)
# QUERY 图像
        list_img = os.listdir(qFolder)
        list_img = [f for f in list_img if '.jpg' in f]
        self.images.extend([os.path.join(qFolder, qIm) for qIm in list_img])
        self.numQ = len(self.images)-self.numDb

        self.positives = None

        qCoordinate = read_gnss(queryGPS)
        ratio_q = len(qCoordinate) / self.numQ
        self.utmQ = np.zeros((self.numQ, 2))
        for i in range(self.numQ):
            self.utmQ[i, :] = np.array(qCoordinate[int(ratio_q*i)-1])

        dCoordinate = read_gnss(databaseGPS)
        ratio_d = len(dCoordinate) / self.numDb
        self.utmDb = np.zeros((self.numDb, 2))
        for i in range(self.numDb):
            self.utmDb[i, :] = np.array(dCoordinate[int(ratio_d*i)-1])

        self.overlap = []
        with open(databaseOverlap) as f:
            for line in f:
                self.overlap.append(int(line))

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        # img = img.resize((896, 224))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        if self.positives is None:
            index = faiss.IndexFlatL2(2)
            index.add(self.utmDb.astype('float32'))
            dis, ind = index.search(self.utmQ.astype('float32'), 629)
            self.positives = []
            for i in range(len(ind)):
                tmp = []
                if self.overlap[i] == 1:
                    for j in ind[i]:
                        c1 = list(self.utmQ[i, :])
                        c2 = list(self.utmDb[j, :])
                        if gnss_distance(c1, c2) < 50:
                            tmp.append(j)
                self.positives.append(tmp)
        falseGT=0
        for p in range(len(self.positives)):
            if self.positives[p] == [] and self.overlap[p] == 1:
                falseGT += 1
        print('falseGT', falseGT)
        return self.positives


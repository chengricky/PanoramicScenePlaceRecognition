import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
import os
from PIL import Image

root_dir = '/media/ricky/Entertainment/YuquanPAL4Localization/ProcessedDataset/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Yuquan dataset')

# the list of database folder (images)
dbFolder = root_dir + 'Afternoon1'
qFolder = root_dir + 'Afternoon2'

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# DB + query
def get_whole_val_set(onlyDB=False):
    return DatasetFromStruct(dbFolder, qFolder,
                                  input_transform=input_transform(),
                                  onlyDB=onlyDB)

class DatasetFromStruct(data.Dataset):
    def __init__(self, dbFolder, qFolder, input_transform=None, onlyDB=False):
        super().__init__()

# DATABASE 图像
        self.input_transform = input_transform
        listImg = os.listdir(dbFolder)
        listImg.sort()
        listImg = listImg[:len(listImg)-1] #get rid of the dir
        self.images = []
        self.images.extend([join(dbFolder, dbIm) for dbIm in listImg])
        self.numDb = len(self.images)
# QUERY 图像
        listImg = os.listdir(qFolder)
        listImg.sort()
        listImg = listImg[:len(listImg) - 1]  # get rid of the dir
        self.images.extend([join(qFolder, qIm) for qIm in listImg])
        self.numQ = len(self.images)-self.numDb

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = img.resize((1024, 213)) #分成5份
        #img = img.resize((896, 224))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)


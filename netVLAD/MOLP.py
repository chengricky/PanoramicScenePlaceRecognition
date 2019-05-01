import torchvision.transforms as transforms
import torch.utils.data as data
from os.path import join, exists
import os
from PIL import Image

root_dir = '/media/ricky/Entertainment/MOLP/summer/evening/route A/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to MOLP dataset')

# the list of database folder (images)
dbFolder = root_dir + 'forward'
qFolder = root_dir + 'backward'

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# DB + query
def get_whole_val_set(panoramicCrop):
    return DatasetFromStruct(dbFolder, qFolder, panoramicCrop, input_transform=input_transform())

class DatasetFromStruct(data.Dataset):
    def __init__(self, dbFolder, qFolder, panoramicCrop, input_transform=None):
        super().__init__()

# DATABASE 图像
        self.input_transform = input_transform
        listImg = os.listdir(dbFolder)
        listImg.sort()
        self.images = []
        self.images.extend([join(dbFolder, dbIm) for dbIm in listImg])
        self.numDb = len(self.images)
# QUERY 图像
        listImg = os.listdir(qFolder)
        listImg.sort()
        self.images.extend([join(qFolder, qIm) for qIm in listImg])
        self.numQ = len(self.images)-self.numDb

        self.crops = int(panoramicCrop)
        self.positives = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = img.resize((224*self.crops, 224))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        if self.positives is None:
            list(range(0, self.numQ))

        return self.positives


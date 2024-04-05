import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from ColorNorm import ColorNorm
import h5py
class ImmuDataset(data.Dataset):
    def __init__(self,csv_file,label_Dict,train=True):
        slideIDx = []
        self.data = pd.read_csv(csv_file)
        self.slides = list(self.data['slide'])
        for i in self.slides:
            slideIDx.append(os.path.basename(i).split('.svs')[0])
        self.labels = [label_Dict['therapy'][i] for i in list(self.data['label'])]
        self.class_sample_count = np.array(
            [len(np.where(self.labels == t)[0]) for t in np.unique(self.labels)])
        self.weights = 1. / self.class_sample_count
        self.weightsDic = {}
        for i in range(len(np.unique(self.labels))):
            self.weightsDic[np.unique(self.labels)[i]] = self.weights[i]
        self.sample_weight = np.array([self.weightsDic[i] for i in self.labels])
        self.slideIDx = slideIDx
        self.sexs = [label_Dict['gender'][i] for i in (self.data['gender'])]
        self.ages = list(self.data['age'])
        #self.sites = list(self.data['site'])
        self.coords = list(self.data['coords'])
        self.features = list(self.data['features'])
        self.train = train


    def __len__(self):
        return len(self.slides)
    def __getitem__(self, index):
        if  torch.is_tensor(index):
            index = index.tolist()
        slideIDx = self.slideIDx[index]
        label = self.labels[index]
        sex = self.sexs[index]
        age = int(self.ages[index])
        #site = self.sites[index]
        if not self.train:
            with h5py.File(self.coords[index,], 'r') as hdf5_file:
                coords = hdf5_file['coords'][:]
            coords = torch.from_numpy(coords)
        with h5py.File(self.features[index], 'r') as hdf5_file:
            print(self.features[index])
            features = hdf5_file['Features'][:]
        features = torch.from_numpy(features)
        if self.train :
            print('This is train model.........')
            return features, label, sex, age,
        else:
            print('This is testing model.........')
            return features,label,sex,age,coords,slideIDx
def main():
    label_Dict = {'gender': {'female': 0, 'male': 1},
                  'therapy': {'Responder': 1, 'Nonresponder': 0}}
    test_dset = ImmuDataset(csv_file='/Data/yangml/Project/TCGA/Public7_TCGA_Part2/lung/data/LUSC/late_stage_lusc_slide/SVS/Trainning/samples_train_dat_latest.csv',label_Dict=label_Dict)#,transform=trans)

    import sys
    sys.path.append(r"/Data/yangml/Important_script/WSIimmune")
    from utils.utils import get_loader
    loader = get_loader(test_dset,training=True,weighted=True)

    i=0
    for batch_idx, (data, label,sex, age) in enumerate(loader):
        i+=1
        if i <2:
            print(data)
            print(label)
            print(sex)
            print(age)
        else:
            break

    dat= next(iter(test_dset))
    i = 0
    for features,label ,sex, age in test_dset:
        i+=1
        if i < 50:
            print("class label:{}".format(label))
            print("sample sex {}".format(sex))
            print("age {}".format(age))
        else:
            break

if __name__ == '__main__':
    main()
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import os
import scipy.io as scio
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy.matlib

class CustomDataset(Dataset):
    def __init__(self, root_dir, ratio=0.8, set_type=None, transform=None): 
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir  

        self.set_type = set_type
        self.transform = transform
        train_dict, val_dict = self.split_data(ratio)

        if self.set_type == 'train':
            self.data, self.label = train_dict['data'], train_dict['label']
        elif self.set_type == 'val':
            self.data, self.label = val_dict['data'], val_dict['label']
        else:
            print("Check Out the Selected Type.")

    def __getitem__(self, item):
        data_path, label_path = self.data[item], self.label[item]
        datafile, labelfile = scio.loadmat(data_path), scio.loadmat(label_path)
        realdata, real_label = datafile['Y'], labelfile['Ylabel']
        real_label = real_label.reshape(1, -1)
        if self.transform:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])

            realdata = transform(realdata).squeeze(0)
            real_label = transform(real_label).squeeze(0)
            
        path_set = data_path.split("/")
        target_num = ord(path_set[-3][3]) - ord('0')
        
        return realdata, real_label[0], target_num

    def __len__(self):
        return len(self.data)

    def split_data(self, ratio):
        train_dataset, val_dataset = [], []  
        train_labelset, val_labelset = [], []

        for data_subroot in os.listdir(self.root_dir):
            data_path = os.path.join(self.root_dir, data_subroot, 'Data')  
            label_path = os.path.join(self.root_dir, data_subroot, 'Label')

            data_file = os.listdir(data_path)
            data_file.sort()
            label_file = os.listdir(label_path)
            label_file.sort()

            train_datafiles, val_datafiles = data_file[:int(len(data_file) * ratio)], \
                                             data_file[int(len(data_file) * ratio):]
            train_labelfiles, val_labelfiles = label_file[:int(len(label_file) * ratio)], \
                                               label_file[int(len(label_file) * ratio):]

            for id in range(len(train_datafiles)):
                train_mat_path, train_label_path = os.path.join(data_path, train_datafiles[id]), \
                                                   os.path.join(label_path, train_labelfiles[id])
                train_dataset.append(train_mat_path)
                train_labelset.append(train_label_path)

            for id in range(len(val_datafiles)):
                val_mat_path, val_label_path = os.path.join(data_path, val_datafiles[id]), \
                                               os.path.join(label_path, val_labelfiles[id])
                val_dataset.append(val_mat_path)
                val_labelset.append(val_label_path)



        train_dict = {'data': train_dataset, 'label': train_labelset}
        val_dict = {'data': val_dataset, 'label': val_labelset}

        return train_dict, val_dict





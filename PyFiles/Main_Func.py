import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import os
import time
import numpy as np
import torchvision
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import scipy.io as scio
from Dataloaders import CustomDataset
from Models import AnmNetwork, CMSELoss


seed_flag = 42  
if seed_flag is not None:
    np.random.seed(seed_flag)
    torch.manual_seed(seed_flag)
    torch.cuda.manual_seed(seed_flag)
    cudnn.deterministic = True


# root_dir = '/home/yuxl/xl_2023/NoiselessSets/' 
root_dir = '/home/yuxl/xl_2023/Snap100File/' 

trainset, valset = CustomDataset(root_dir=root_dir, set_type='train', transform=True),\
                   CustomDataset(root_dir=root_dir, set_type='val', transform=True)

epochs = 500
batch_size = 20
train_loader, val_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True),\
                           DataLoader(dataset=valset, batch_size=batch_size, shuffle=True)
model = AnmNetwork(K=20) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.load(model_path)
# device = torch.device('cpu')
model.to(device)

criterion = CMSELoss("sum").to(device) 
learning_rate = 1e-1
weight_decay = 5e-4  
lr_gamma = 0.01
lr_decay = 0.75
overfitting_threshold = 100
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x: learning_rate * (1. + lr_gamma * float(x)) ** (-lr_decay))


def train(dataloader, model, criterion, optimizer, lr_scheduler, hist, device):
    data_iter = iter(dataloader)

    avg_regression_loss = 0
    model.train()
    for batch_idx in range(len(dataloader)):
        data, label, num_set = next(data_iter)
        data, label = Variable(data.to(device)), Variable(label.to(device))
        
        T_new, uvec = model(data)
        regression_loss = criterion(uvec, label)
        optimizer.zero_grad()
        regression_loss.requires_grad_(True)
        avg_regression_loss += regression_loss.item()

        regression_loss.backward()
        optimizer.step()
        lr_scheduler.step()
    
    avg_regression_loss /= len(dataloader)

    hist['train_loss'].append(avg_regression_loss)
    print('Train Loss: {:.4f}'.format(avg_regression_loss))


def valid(dataloader, model, criterion, hist, device):
    data_iter = iter(dataloader)
    avg_regression_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx in range(len(dataloader)):

            data, label, num_set = next(data_iter)
            data, label = Variable(data.to(device)), Variable(label.to(device))

            T_new, uvec = model(data)
            regression_loss = criterion(uvec, label)
            avg_regression_loss += regression_loss.item()

        avg_regression_loss /= len(dataloader)
        hist['val_loss'].append(avg_regression_loss)
        print('Valid Loss: {:.4f}'.format(avg_regression_loss))


def load_model(save_path):
    model_path = os.path.join(save_path, 'model.pth')
    model = torch.load(model_path)

    para_path = os.path.join(save_path, 'para.mat')
    paraset = []
    for param in model.parameters():
        data = param.data.cpu().numpy()
        paraset.append(data[0])

    print(paraset)

    scio.savemat(para_path, {'Para': paraset})

def main(save_path):
    count_overfiiting = 0
    min_loss = 1e2
    train_hist = {'train_loss':[]}
    val_hist = {'val_loss': []}
    for epoch in range(epochs):
        t0 = time.time()
        print("Epoch:{}".format(epoch))
        train(train_loader, model, criterion, optimizer, lr_scheduler, train_hist, device)

        t1 = time.time() - t0

        valid(val_loader, model, criterion, val_hist, device)
        print("Training Time Cost:{:.4f}s".format(t1))

        if val_hist['val_loss'][-1] < min_loss:
            min_loss = val_hist['val_loss'][-1]
            print("----- Min Loss : {:.4f} -----".format(min_loss))
            torch.save(model, save_path + 'model.pth')
            count_overfiiting = 0
        else:
            count_overfiiting += 1

        if count_overfiiting == overfitting_threshold:
            break
            
    train_mat_path = os.path.join(save_path, 'train_loss.mat')
    val_mat_path = os.path.join(save_path, 'val_loss.mat')
    scio.savemat(train_mat_path, {'Y': train_hist})
    scio.savemat(val_mat_path, {'Y': val_hist})


if __name__ == '__main__':
    save_path = '/home/yuxl/xl_2023/DeepUnfolding/Snap100_logs2/'
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    main(save_path)
    load_model(save_path)














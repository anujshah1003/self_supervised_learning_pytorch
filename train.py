import os,sys
import numpy as np
from tqdm import tqdm
from itertools import islice
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader 
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter


from PIL import Image
import matplotlib.pyplot as plt

#from lshash.lshash import LSHash

import utils
import models
import dataloaders
from evaluate import validate,test

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

torch.set_default_tensor_type(dtype)
#%%

def train(epoch, model, device, dataloader, optimizer, exp_lr_scheduler, criterion, experiment_dir, writer):
    """ Train loop, predict rotations. """
    loss_record = utils.RunningAverage()
    acc_record = utils.RunningAverage()
    correct=0
    total=0
    save_path = experiment_dir + '/'
    os.makedirs(save_path, exist_ok=True)
    model.train()
    for batch_idx, (data,label,_,_) in enumerate(tqdm(islice(dataloader,10))):
#    for batch_idx, (data, label, _,_) in enumerate(tqdm(dataloader)):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)

        # measure accuracy and record loss
        confidence, predicted = output.max(1)
        correct += predicted.eq(label).sum().item()
        #acc = utils.compute_acc(output, label)
        total+=label.size(0)
        acc = correct/total
        
        acc_record.update(100*acc)
        loss_record.update(loss.item())

        writer.add_scalar('Loss/train', loss.item(), epoch + batch_idx)
        writer.add_scalar('Acc/train', loss.item(), epoch + batch_idx)

        print('Train Step: {}/{} Loss: {:.4f}; Acc: {:.4f}'.format(batch_idx,len(dataloader), loss_record(), acc_record()))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if exp_lr_scheduler:  
        exp_lr_scheduler.step()

    writer.add_scalar('Loss_epoch/train', loss_record(), epoch)
    writer.add_scalar('Acc_epoch/train', acc_record(), epoch)
    print('Train Epoch: {} Avg Loss: {:.4f}; Avg Acc: {:.4f}'.format(epoch, loss_record(), acc_record()))

    return loss_record,acc_record

#%%  
def train_and_evaluate(params):
    
    # Training settings
    experiment_dir = os.path.join('experiments',params.save_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    use_cuda = params.use_cuda and torch.cuda.is_available()
    params.use_cuda=use_cuda
    device = torch.device("cuda:{}".format(params.cuda_num) if use_cuda else "cpu")
    # initialize the tensorbiard summary writer
    writer = SummaryWriter(experiment_dir + '/tboard' )

    ## get the dataloaders
    dloader_train,dloader_test = dataloaders.get_dataloaders(params)
    
    # Load the model
    model = models.get_model(params)
    model = model.to(device)

    images,_ ,_,_ = next(iter(dloader_train))
    images = images.to(device)
    writer.add_graph(model, images)

    # follow the same setting as RotNet paper
    optimizer = optim.SGD(model.parameters(), lr=float(params.lr), momentum=float(params.momentum), weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    best_loss = 1000
    for epoch in range(params.num_epochs + 1):
        
        print('\nTrain for Epoch: {}/{}'.format(epoch,params.num_epochs))
        train_loss,train_acc = train(epoch, model, device, dloader_train, optimizer, exp_lr_scheduler, criterion, experiment_dir, writer)
        
        # validate after every epoch
        print('\nValidate for Epoch: {}/{}'.format(epoch,params.num_epochs))
        val_loss,val_acc = validate(epoch, model, device, dloader_test, criterion, experiment_dir, writer)

        is_best = val_loss() < best_loss
        best_loss = min(val_loss(), best_loss)
        if epoch % params.save_intermediate_weights==0:
            utils.save_checkpoint({'Epoch': epoch,'state_dict': model.state_dict(),
                                   'optim_dict' : optimizer.state_dict()}, 
                                    is_best, experiment_dir, checkpoint='{}_{}rot_epoch{}_checkpoint.pth'.format( params.network.lower(), str(params.num_rot),str(epoch)),\
                                    
                                    best_model='{}_{}rot_epoch{}_best.pth'.format(params.network.lower(), str(params.num_rot),str(epoch))
                                    )
    writer.close()
    
    print('\nEvaluate on Test')
    test_loss,test_acc = test(model, device, dloader_test, criterion, experiment_dir)
    
    # save the configuration file within that experiment directory
    utils.save_yaml(params,save_path=os.path.join(experiment_dir,'config.yaml'))
    
if __name__=='__main__':
    config_file='config/config.yaml'
    cfg = utils.load_yaml(config_file,config_type='object')
    train_and_evaluate(cfg)
    
'''

OSError: [Errno 24] Too many open files: 'D:\\2020\\Trainings\\self-supervised-learning\\dataset\\flowers\\tulip\\tulip_000826.png'
to solve this

'''

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
import logging

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

def train(epoch, model, device, dataloader, optimizer, scheduler, criterion, experiment_dir, writer):
    """ Train loop, predict rotations. """
    global iter_cnt
#    progbar = tqdm(total=len(dataloader), desc='Train')
    progbar = tqdm(total=10, desc='Train')

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

        writer.add_scalar('train/Loss_batch', loss.item(), iter_cnt)
        writer.add_scalar('train/Acc_batch', acc, iter_cnt)
        iter_cnt+=1

#        logging.info('Train Step: {}/{} Loss: {:.4f}; Acc: {:.4f}'.format(batch_idx,len(dataloader), loss_record(), acc_record()))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progbar.set_description('Train (loss=%.4f)' % (loss_record()))
        progbar.update(1)
        
    if scheduler:  
        scheduler.step()
        
    LR=optimizer.param_groups[0]['lr']


    writer.add_scalar('train/Loss_epoch', loss_record(), epoch)
    writer.add_scalar('train/Acc_epoch', acc_record(), epoch)
    logging.info('Train Epoch: {} LR: {:.4f} Avg Loss: {:.4f}; Avg Acc: {:.4f}'.format(epoch,LR, loss_record(), acc_record()))

    return loss_record,acc_record

#%%  
def train_and_evaluate(cfg):
    
    #Training settings
    experiment_dir = os.path.join('experiments',cfg.exp_type,cfg.save_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        
    utils.set_logger(os.path.join(experiment_dir,cfg.log))
    logging.info('-----------Starting Experiment------------')
    use_cuda = cfg.use_cuda and torch.cuda.is_available()
    cfg.use_cuda=use_cuda
    device = torch.device("cuda:{}".format(cfg.cuda_num) if use_cuda else "cpu")
    # initialize the tensorbiard summary writer
    writer = SummaryWriter(experiment_dir + '/tboard' )

    ## get the dataloaders
    dloader_train,dloader_val,dloader_test = dataloaders.get_dataloaders(cfg)
    
    # Load the model
    model = models.get_model(cfg)
    
    if cfg.use_pretrained:
        ssl_exp_dir = experiment_dir = os.path.join('experiments',\
                                        'self-supervised',cfg.ssl_pretrained_exp_path)
        state_dict = torch.load(os.path.join(ssl_exp_dir,cfg.ssl_weight),\
                                map_location=device)
        # the stored dict has 3 informations - epoch,state_dict and optimizer
        state_dict=state_dict['state_dict']
        del state_dict['fc.weight']
        del state_dict['fc.bias']
    
        model.load_state_dict(state_dict, strict=False)
    
        # Only finetune fc layer
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    model = model.to(device)

    images,_ ,_,_ = next(iter(dloader_train))
    images = images.to(device)
    writer.add_graph(model, images)

    # follow the same setting as RotNet paper
    optimizer = optim.SGD(model.parameters(), lr=float(cfg.lr), momentum=float(cfg.momentum), weight_decay=5e-4, nesterov=True)
    if cfg.scheduler:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
    else:
        scheduler=None
    criterion = nn.CrossEntropyLoss()
    
    global iter_cnt
    iter_cnt=0
    best_loss = 1000
    for epoch in range(cfg.num_epochs):
        
#        print('\nTrain for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        logging.info('\nTrain for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        train_loss,train_acc = train(epoch, model, device, dloader_train, optimizer, scheduler, criterion, experiment_dir, writer)
        
        # validate after every epoch
#        print('\nValidate for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        logging.info('\nValidate for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        val_loss,val_acc = validate(epoch, model, device, dloader_val, criterion, experiment_dir, writer)
        logging.info('Val Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, val_loss, val_acc))
        
        for name, weight in model.named_parameters():
            writer.add_histogram(name,weight, epoch)
            writer.add_histogram(f'{name}.grad',weight.grad, epoch)
            
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if epoch % cfg.save_intermediate_weights==0:
            utils.save_checkpoint({'Epoch': epoch,'state_dict': model.state_dict(),
                                   'optim_dict' : optimizer.state_dict()}, 
                                    is_best, experiment_dir, checkpoint='{}_{}rot_epoch{}_checkpoint.pth'.format( cfg.network.lower(), str(cfg.num_rot),str(epoch)),\
                                    
                                    best_model='{}_{}rot_epoch{}_best.pth'.format(cfg.network.lower(), str(cfg.num_rot),str(epoch))
                                    )
    writer.close()
    
#    print('\nEvaluate on test')
    logging.info('\nEvaluate on test')
    test_loss,test_acc = test(model, device, dloader_test, criterion, experiment_dir)
    logging.info('Test: Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(test_loss, test_acc))

    # save the configuration file within that experiment directory
    utils.save_yaml(cfg,save_path=os.path.join(experiment_dir,'config_sl.yaml'))
    logging.info('-----------End of Experiment------------')
#%%
if __name__=='__main__':
    config_file='config/config_sl.yaml'
    cfg = utils.load_yaml(config_file,config_type='object')
    train_and_evaluate(cfg)
    
'''

OSError: [Errno 24] Too many open files: 'D:\\2020\\Trainings\\self-supervised-learning\\dataset\\flowers\\tulip\\tulip_000826.png'
to solve this

'''


import utils
from tqdm import tqdm
from itertools import islice
import torch
import os
import dataloaders
import models
import torch.nn as nn


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

torch.set_default_tensor_type(dtype)

#%%

def validate(epoch, model, device, dataloader, criterion, args, writer):
    """ Test loop, print metrics """
    progbar = tqdm(total=len(dataloader), desc='Val')

    
    loss_record = utils.RunningAverage()
    acc_record = utils.RunningAverage()
    model.eval()
    with torch.no_grad():
    #    for batch_idx, (data,label,_,_) in enumerate(tqdm(islice(dataloader,10))):
        for batch_idx, (data, label,_,_) in enumerate(tqdm(dataloader)):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
    
            # measure accuracy and record loss
            acc = utils.compute_acc(output, label)
    #        acc_record.update(100 * acc[0].item())
            acc_record.update(100*acc[0].item()/data.size(0))
            loss_record.update(loss.item())
            #print('val Step: {}/{} Loss: {:.4f} \t Acc: {:.4f}'.format(batch_idx,len(dataloader), loss_record(), acc_record()))
            progbar.set_description('Val (loss=%.4f)' % (loss_record()))
            progbar.update(1)

    writer.add_scalar('validation/Loss_epoch', loss_record(), epoch)
    writer.add_scalar('validation/Acc_epoch', acc_record(), epoch)
    
    return loss_record(),acc_record()

def test( model, device, dataloader, criterion, args):
    """ Test loop, print metrics """
    loss_record = utils.RunningAverage()
    acc_record = utils.RunningAverage()
    model.eval()
    with torch.no_grad():
     #   for batch_idx, (data,label,_,_) in enumerate(tqdm(islice(dataloader,10))):
        for batch_idx, (data, label,_,_) in enumerate(tqdm(dataloader)):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
    
            # measure accuracy and record loss
            acc = utils.compute_acc(output, label)
    #        acc_record.update(100 * acc[0].item())
            acc_record.update(100*acc[0].item()/data.size(0))
            loss_record.update(loss.item())
#            print('Test Step: {}/{} Loss: {:.4f} \t Acc: {:.4f}'.format(batch_idx,len(dataloader), loss_record(), acc_record()))

    return loss_record(),acc_record()
#%%
if __name__=='__main__':
    
    experiment_dir = r'D:\2020\Trainings\self_supervised_learning\experiments\supervised\sl_rotnet_pretrain_finetune_4'
    config_file=os.path.join(experiment_dir,'config_sl.yaml')
    ckpt_name='resnet18_best.pth'
    ckpt_path=os.path.join(experiment_dir,ckpt_name)
    
    assert os.path.isfile(config_file), "No parameters config file found at {}".format(config_file)

    cfg = utils.load_yaml(config_file,config_type='object')

    use_cuda = cfg.use_cuda and torch.cuda.is_available()
    cfg.use_cuda=use_cuda
    device = torch.device("cuda:{}".format(cfg.cuda_num) if use_cuda else "cpu")


    ## get the dataloaders
    _,_,dloader_test = dataloaders.get_dataloaders(cfg,val_split=.2)
    
    
    # Load the model
    model = models.get_model(cfg)
    state_dict = torch.load(ckpt_path,map_location=device)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    test_loss,test_acc = test(model, device, dloader_test, criterion, experiment_dir)

    print('Test: Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(test_loss, test_acc))

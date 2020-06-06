import utils
from tqdm import tqdm
from itertools import islice
import torch

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

torch.set_default_tensor_type(dtype)

#%%

def validate(epoch, model, device, dataloader, criterion, args, writer):
    """ Test loop, print metrics """
    loss_record = utils.RunningAverage()
    acc_record = utils.RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch_idx, (data,label,_,_) in enumerate(tqdm(islice(dataloader,10))):
    #    for batch_idx, (data, label,_,_) in enumerate(tqdm(dataloader)):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
    
            # measure accuracy and record loss
            acc = utils.compute_acc(output, label)
    #        acc_record.update(100 * acc[0].item())
            acc_record.update(100*acc[0].item()/data.size(0))
            loss_record.update(loss.item())
            print('val Step: {}/{} Loss: {:.4f} \t Acc: {:.4f}'.format(batch_idx,len(dataloader), loss_record(), acc_record()))


    writer.add_scalar('Loss/validation', loss_record(), epoch)
    writer.add_scalar('Accuracy/validation', acc_record(), epoch)
    
    print('Val Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record(), acc_record()))
    return acc_record,loss_record

def test( model, device, dataloader, criterion, args):
    """ Test loop, print metrics """
    loss_record = utils.RunningAverage()
    acc_record = utils.RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch_idx, (data,label,_,_) in enumerate(tqdm(islice(dataloader,10))):
    #    for batch_idx, (data, label,_,_) in enumerate(tqdm(dataloader)):
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
    
            # measure accuracy and record loss
            acc = utils.compute_acc(output, label)
    #        acc_record.update(100 * acc[0].item())
            acc_record.update(100*acc[0].item()/data.size(0))
            loss_record.update(loss.item())
            print('Test Step: {}/{} Loss: {:.4f} \t Acc: {:.4f}'.format(batch_idx,len(dataloader), loss_record(), acc_record()))

    print('Test: Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(loss_record(), acc_record()))
    return loss_record,acc_record
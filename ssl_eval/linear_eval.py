'''
A script to evaluate the pretrained representations on downstream task using
logistic regression

'''
import os,sys
sys.path.append(os.path.join(os.path.dirname("__file__"),'..'))
import numpy as np
from tqdm import tqdm
from itertools import islice
import yaml
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.decomposition import PCA,KernelPCA


from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter


import pickle

#from lshash.lshash import LSHash
import logging

import utils

#from evaluate import validate,test

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

torch.set_default_tensor_type(dtype)
#%%
def get_pca_feats(feat,num_dim=400):
    pca = PCA()
    #kpca = KernelPCA()

    feat_pca = pca.fit_transform(feat)
    #x_pca = pd.DataFrame(x_pca)
    #x_pca.head()

    explained_variance = pca.explained_variance_ratio_
    #explained_variance
    np.sum(explained_variance[0:num_dim])
    return(feat_pca[:,0:num_dim])

def get_data(args,feature_path):
        

    train_feat_path = os.path.join(feature_path,'train_features_dict.p')
    test_feat_path = os.path.join(feature_path,'test_features_dict.p')
    train_label_path = os.path.join(feature_path,'train_labels_dict.p')
    test_label_path = os.path.join(feature_path,'test_labels_dict.p')
    
    assert os.path.isfile(train_feat_path), "No features dictionary found at {}".format(train_feat_path)
    assert os.path.isfile(test_feat_path), "No features dictionary found at {}".format(test_feat_path)

    assert os.path.isfile(train_label_path), "No labels dictionary found at {}".format(train_label_path)
    assert os.path.isfile(test_label_path), "No labels dictionary found at {}".format(test_label_path)
    
    train_feature_dict = pickle.load(open(train_feat_path,'rb'))
    test_feature_dict = pickle.load(open(test_feat_path,'rb'))
    
    train_label_dict = pickle.load(open(train_label_path,'rb'))
    test_label_dict = pickle.load(open(test_label_path,'rb'))
    
    labels=[]
    for img in train_feature_dict.keys():
        label=train_label_dict[img]
        labels.append(int(label))
    
    x_tr = np.array(list(train_feature_dict.values())) 
    y_tr = np.array(labels)
    if cfg.pca_dim:
        x_tr=get_pca_feats(x_tr,num_dim=cfg.pca_dim)
    
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.18, random_state=42)
    
    del train_feature_dict        
    del train_label_dict 
    
    labels=[]
    for img in test_feature_dict.keys():
        label=test_label_dict[img]
        labels.append(int(label))

    x_ts = np.array(list(test_feature_dict.values())) 
    y_ts = np.array(labels) 
    if cfg.pca_dim:
        x_ts=get_pca_feats(x_ts,num_dim=cfg.pca_dim)

    del test_feature_dict   
    del test_label_dict 
    del labels
    
    print ('number of tr samples: {}'.format(x_tr.shape[0]))
    print ('number of val samples: {}'.format(x_val.shape[0]))
    print ('number of test samples: {}'.format(x_ts.shape[0]))

    return (x_tr,y_tr,x_val,y_val,x_ts,y_ts)
        

class LogisticRegression(nn.Module):
    
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)       

def normalize_dataset(X_train,X_val, X_test):
    print("Standard Scaling Normalizer")
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train,X_val,X_test

def get_data_loaders(cfg, X_train, y_train,X_val,y_val, X_test, y_test):
    if cfg.normalize:
        X_train,X_val, X_test = normalize_dataset(X_train,X_val, X_test)

    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).type(torch.long))
    train_loader = torch.utils.data.DataLoader(train, batch_size=cfg.batch_size, shuffle=False)
    
    val = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).type(torch.long))
    val_loader = torch.utils.data.DataLoader(val, batch_size=cfg.batch_size, shuffle=False)
    
    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).type(torch.long))
    test_loader = torch.utils.data.DataLoader(test, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader,test_loader

def train(epoch,model,device,dataloader,optimizer,scheduler,criterion,experiment_dir, writer):
    
    """ Train loop, predict rotations. """
    global iter_cnt
    progbar = tqdm(total=len(dataloader), desc='Train')
#    progbar = tqdm(total=10, desc='Train')

    loss_record = utils.RunningAverage()
    acc_record = utils.RunningAverage()
    correct=0
    total=0
    save_path = experiment_dir + '/'
    os.makedirs(save_path, exist_ok=True)
    model.train()
 #   for batch_idx,(data,label) in enumerate(tqdm(islice(dataloader,10))):
    for batch_idx, (data, label) in enumerate(tqdm(dataloader)):
        data, label = data.to(device), label.to(device)
        #optimizer.zero_grad()
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
#    logging.info('Train Epoch: {} LR: {:.4f} Avg Loss: {:.4f}; Avg Acc: {:.4f}'.format(epoch,LR, loss_record(), acc_record()))

    return loss_record(),acc_record()

def validate(epoch, model, device, dataloader, criterion, args, writer):
    """ Test loop, print metrics """
    progbar = tqdm(total=len(dataloader), desc='Val')

    
    loss_record = utils.RunningAverage()
    acc_record = utils.RunningAverage()
    model.eval()
    with torch.no_grad():
    #    for batch_idx, (data,label,_,_) in enumerate(tqdm(islice(dataloader,10))):
        for batch_idx, (data, label) in enumerate(tqdm(dataloader)):
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
        for batch_idx, (data, label) in enumerate(tqdm(dataloader)):
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

def train_and_evaluate(cfg,dloader_train,dloader_val,dloader_test,device,writer,experiment_dir):
    
    if cfg.opt=='adam':
        optimizer = optim.Adam(model.parameters(), lr=float(cfg.lr))#, momentum=float(cfg.momentum), weight_decay=5e-4, nesterov=True)
    elif cfg.opt=='sgd':
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
        logging.info('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, train_loss, train_acc))
        
        # validate after every epoch
#        print('\nValidate for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        logging.info('\nValidate for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        val_loss,val_acc = validate(epoch, model, device, dloader_val, criterion, experiment_dir, writer)
        logging.info('Val Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, val_loss, val_acc))
        
       # for name, weight in model.named_parameters():
        #    writer.add_histogram(name,weight, epoch)
         #   writer.add_histogram(f'{name}.grad',weight.grad, epoch)
            
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        if epoch % cfg.save_intermediate_weights==0 or is_best:
            utils.save_checkpoint({'Epoch': epoch,'state_dict': model.state_dict(),
                                   'optim_dict' : optimizer.state_dict()}, 
                                    is_best, experiment_dir, checkpoint='{}_epoch{}_checkpoint.pth'.format( cfg.network.lower(),str(epoch)),\
                                    
                                    best_model='{}_best.pth'.format(cfg.network.lower())
                                    )
    
#    print('\nEvaluate on test')
    logging.info('\nEvaluate test result on best ckpt')
    state_dict = torch.load(os.path.join(experiment_dir,'{}_best.pth'.format(cfg.network.lower())),\
                                map_location=device)
    model.load_state_dict(state_dict['state_dict'], strict=False)

    test_loss,test_acc = test(model, device, dloader_test, criterion, experiment_dir)
    logging.info('Test: Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(test_loss, test_acc))
    writer.add_text('test performance on best ckpt','test_loss {}; test_acc {}'.format(test_loss,test_acc))
    writer.close()

    # save the configuration file within that experiment directory
    utils.save_yaml(cfg,save_path=os.path.join(experiment_dir,'config_linear.yaml'))
    logging.info('-----------End of Experiment------------')

 
#%%           
if __name__=='__main__':
    
    config_file='../config/config_linear.yaml'
    cfg = utils.load_yaml(config_file,config_type='object')
    feature_dir=os.path.join(cfg.root_path,'experiments',cfg.exp_type,\
                                   cfg.feat_extract_exp_dir,cfg.features)
    
    X_tr,y_tr,X_val,y_val,X_ts,y_ts = get_data(cfg,feature_dir)
    
    #Training settings
    experiment_dir = os.path.join(cfg.root_path,'experiments',cfg.exp_type,\
                                  cfg.feat_extract_exp_dir,cfg.save_dir,cfg.features)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        
    utils.set_logger(os.path.join(experiment_dir,cfg.log))
    logging.info('-----------Starting Experiment------------')
    use_cuda = cfg.use_cuda and torch.cuda.is_available()
    cfg.use_cuda=use_cuda
    device = torch.device("cuda:{}".format(cfg.cuda_num) if use_cuda else "cpu")
    # initialize the tensorbiard summary writer
#    writer = SummaryWriter(experiment_dir + '/tboard' )
    logs=os.path.join(cfg.root_path,'experiments',cfg.exp_type,\
                      cfg.feat_extract_exp_dir,cfg.save_dir,'tboard_linear')
    writer = SummaryWriter(logs + '/lin_probe_{}'.format(cfg.features) )
    
    logging.info('load the data and the model ........')
    model = LogisticRegression(n_features=X_tr.shape[1], n_classes=5)
    dloader_train,dloader_val,dloader_test = get_data_loaders(cfg,X_tr, y_tr,X_val,y_val, X_ts, y_ts)

    train_and_evaluate(cfg,dloader_train,dloader_val,dloader_test,device,writer,experiment_dir)
    
    
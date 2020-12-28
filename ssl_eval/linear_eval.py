'''
A script to evaluate the pretrained representations on downstream task using
logistic regression

'''
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

def train(epoch, logistic_model,ssl_model, device, dataloader, optimizer, scheduler, criterion, experiment_dir, writer):
   
    """ Train loop, predict rotations. """
    loss_record = utils.RunningAverage()
    acc_record = utils.RunningAverage()
    correct=0
    total=0
    save_path = experiment_dir + '/'
    os.makedirs(save_path, exist_ok=True)
    logistic_model.train()
    
    for batch_idx, (data,label,_,_) in enumerate(tqdm(islice(dataloader,10))):
#    for batch_idx, (data, label, _,_) in enumerate(tqdm(dataloader)):
        data, label = data.to(device), label.to(device)
        
         # get encoding
        with torch.no_grad():
            feat_vec = ssl_model(data)
        
        output = logistic_model(feat_vec)
        
        loss = criterion(output, label)

        # measure accuracy and record loss
        confidence, predicted = output.max(1)
        correct += predicted.eq(label).sum().item()
        #acc = utils.compute_acc(output, label)
        total+=label.size(0)
        acc = correct/total
        
        acc_record.update(100*acc)
        loss_record.update(loss.item())

        writer.add_scalar('Loss/train_logistic', loss.item(), epoch + batch_idx)
        writer.add_scalar('Acc/train_logistic', loss.item(), epoch + batch_idx)

#        logging.info('Train Step: {}/{} Loss: {:.4f}; Acc: {:.4f}'.format(batch_idx,len(dataloader), loss_record(), acc_record()))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if scheduler:  
        scheduler.step()
    LR=optimizer.param_groups[0]['lr']

    writer.add_scalar('Loss_epoch/train_logistic', loss_record(), epoch)
    writer.add_scalar('Acc_epoch/train_logistic', acc_record(), epoch)
    logging.info('Train Epoch: {} LR: {:.5f} Avg Loss: {:.4f}; Avg Acc: {:.4f}'.format(epoch,LR, loss_record(), acc_record()))

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
    dloader_train,dloader_val,dloader_test = dataloaders.get_dataloaders(cfg,val_split=.2)
    
    # Load the model
    model = models.get_model(cfg)
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

    best_loss = 1000
    for epoch in range(cfg.num_epochs + 1):
        
#        print('\nTrain for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        logging.info('\nTrain for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        train_loss,train_acc = train(epoch, model, device, dloader_train, optimizer, scheduler, criterion, experiment_dir, writer)
        
        # validate after every epoch
#        print('\nValidate for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        logging.info('\nValidate for Epoch: {}/{}'.format(epoch,cfg.num_epochs))
        val_loss,val_acc = validate(epoch, model, device, dloader_val, criterion, experiment_dir, writer)
        logging.info('Val Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, val_loss, val_acc))

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
    utils.save_yaml(cfg,save_path=os.path.join(experiment_dir,'config_ssl.yaml'))
    logging.info('-----------End of Experiment------------')
  

#%%

def get_data(args):
        
    train_feat_path = os.path.join(args.experiment_path,'train_features_dict.p')
    val_feat_path = os.path.join(args.experiment_path,'val_features_dict.p')
    test_feat_path = os.path.join(args.experiment_path,'test_features_dict.p')

    assert os.path.isfile(train_feat_path), "No features dictionary found at {}".format(train_feat_path)
    assert os.path.isfile(val_feat_path), "No features dictionary found at {}".format(val_feat_path)
    assert os.path.isfile(test_feat_path), "No features dictionary found at {}".format(test_feat_path)
    
    train_feature_dict = pickle.load(open(train_feat_path,'rb'))
    val_feature_dict = pickle.load(open(val_feat_path,'rb'))
    test_feature_dict = pickle.load(open(test_feat_path,'rb'))
    
    print ('number of tr samples: {}'.format(len(train_feature_dict)))
    print ('number of val samples: {}'.format(len(val_feature_dict)))
    print ('number of test samples: {}'.format(len(test_feature_dict)))

    train_annotation = pd.read_csv(os.path.join(args.root_data_path,'train', 'train_labels.csv'))
    val_annotation = pd.read_csv(os.path.join(args.root_data_path,'val', 'val_labels.csv'))
    test_annotation = pd.read_csv(os.path.join(args.root_data_path,'test', 'test_labels.csv'))
    
    labels=[]
    for img in train_feature_dict.keys():
        img_name = img.split('\\')[-1]
        label = train_annotation[train_annotation['img_name']==img_name]['class']
        labels.append(int(label))
    
    x_tr = np.array(list(train_feature_dict.values())) 
    y_tr = np.array(labels)
    
    labels=[]           
    for img,feat in val_feature_dict.items():
        img_name = img.split('\\')[-1]
        label = val_annotation[val_annotation['img_name']==img_name]['class']
        labels.append(int(label))
        
    x_val = np.array(list(val_feature_dict.values())) 
    y_val = np.array(labels)
    
    labels=[]           
    for img,feat in test_feature_dict.items():
        img_name = img.split('\\')[-1]
        label = test_annotation[test_annotation['img_name']==img_name]['class']
        labels.append(int(label))

    x_ts = np.array(list(test_feature_dict.values())) 
    y_ts = np.array(labels) 
    
    del train_feature_dict   
    del val_feature_dict   
    del test_feature_dict   
    del labels
    
    return (x_tr,y_tr,x_val,y_val,x_ts,y_ts)
        

class LogisticRegression(nn.Module):
    
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)       

class LogiticRegressionEvaluator(object):
    def __init__(self, n_features, n_classes,args):
        self.args = args
        self.device = args.device
        self.log_regression = LogisticRegression(n_features, n_classes).to(self.device)
        self.scaler = preprocessing.StandardScaler()

        
    def _normalize_dataset(self, X_train,X_val, X_test):
        print("Standard Scaling Normalizer")
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        return X_train,X_val,X_test

    @staticmethod
    def _sample_weight_decay():
        # We selected the l2 regularization parameter from a range of 45 logarithmically spaced values between 10−6 and 105
        weight_decay = np.logspace(-6, 5, num=45, base=10.0)
        weight_decay = np.random.choice(weight_decay)
        print("Sampled weight decay:", weight_decay)
        return weight_decay

    def eval(self, test_loader):
        correct = 0
        total = 0
        
        with torch.no_grad():
          self.log_regression.eval()
          for batch_x, batch_y in tqdm(test_loader):
              batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)
              logits = self.log_regression(batch_x)
        
              predicted = torch.argmax(logits, dim=1)
              total += batch_y.size(0)
              correct += (predicted == batch_y).sum().item()
        
          final_acc = 100 * correct / total
          self.log_regression.train()
          return final_acc


    def get_data_loaders(self, X_train, y_train,X_val,y_val, X_test, y_test):
        X_train,X_val, X_test = self._normalize_dataset(X_train,X_val, X_test)
    
        train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).type(torch.long))
        train_loader = torch.utils.data.DataLoader(train, batch_size=self.args.bs, shuffle=False)
        
        val = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).type(torch.long))
        val_loader = torch.utils.data.DataLoader(val, batch_size=self.args.bs, shuffle=False)
        
        test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).type(torch.long))
        test_loader = torch.utils.data.DataLoader(test, batch_size=self.args.bs, shuffle=False)
        return train_loader, val_loader,test_loader

    def train(self, train_loader, val_loader):
        
        weight_decay = self._sample_weight_decay()
    
        optimizer = torch.optim.Adam(self.log_regression.parameters(), 1e-3)#, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
    
        best_accuracy = 0
    
        for e in range(self.args.epochs):
            logging.info('\n Training for epoch: {}'.format(e))
            for batch_x, batch_y in tqdm(train_loader):
    
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        
                optimizer.zero_grad()
        
                logits = self.log_regression(batch_x)
        
                loss = criterion(logits, batch_y)
        
                loss.backward()
                optimizer.step()
                
            logging.info('validating for epoch {}'.format(e))
            epoch_acc = self.eval(val_loader)
            logging.info('val Acc for epoch {}: {}'.format(e,epoch_acc))
          
            if epoch_acc > best_accuracy:
                #print("Saving new model with accuracy {}".format(epoch_acc))
                best_accuracy = epoch_acc
                save_location = os.path.join(self.args.experiment_path,'log_regression.pth')
                torch.save(self.log_regression.state_dict(), save_location)
    
        logging.info("-------------Training Complete--------------")
        logging.info("Best accuracy for val data:", best_accuracy)
 
#%%           
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root_data_path', default=r'D:\2020\project_small_data\Tire_inspection\tire_inspection_cropped_data_final', help="Directory containing the dataset")

    parser.add_argument('--experiment_path', type=str, default='exp_1',
                        help='the name of the experiment (dir where all the \
                        log files and trained weights of the experimnet will be saved)')
   
    parser.add_argument('--epochs', type=int, default=50,
                        help='the number of epochs')
    
    parser.add_argument('--bs', type=int, default=64,
                        help='the batch size')
    
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu or cuda to use')
    
    parser.add_argument('--log_file', default='lin_eval.log', help="name of your log file, eg- train.log")


    args = parser.parse_args()
        
    if torch.cuda.is_available():
        args.device = torch.device("cuda:2")
    else:
        args.device = "cpu"

    
    args.experiment_path = r'D:\2020\project_small_data\Small_Data\pretraining\experiment_dir\exp_1'
    
    X_tr,y_tr,X_val,y_val,X_ts,y_ts = get_data(args)
    
    utils.set_logger(os.path.join(args.experiment_path, args.log_file))

    logging.info('load the data and the model ........')
    log_regressor_evaluator = LogiticRegressionEvaluator(n_features=X_tr.shape[1], n_classes=2,args=args)

    tr_loader,val_loader,ts_loader = log_regressor_evaluator.get_data_loaders(X_tr, y_tr,X_val,y_val, X_ts, y_ts)

    logging.info('start training ........')
    log_regressor_evaluator.train(tr_loader, val_loader)
    
    logging.info('Evaluate on test data ........')
    test_acc = log_regressor_evaluator.eval(ts_loader)
    logging.info('Test Data Acc: {}'.format(test_acc))

    
#    train_loader,val_loader,test_loader

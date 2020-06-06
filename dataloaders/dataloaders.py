'''


A script for loading the data and serving it to the model for pretraining

'''
import os,sys
sys.path.append(os.path.join(os.path.dirname("__file__"),'.'))

#Apparently 512 is the maximum in python. I found the solution here- https://stackoverflow.com/a/28212496/8875017

#import win32file
#win32file._setmaxstdio(10000)

from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 

from dataloaders.flowers_dataset import FlowersDataset,rotnet_collate_fn
#from utils.transformations import TransformsSimCLR
import utils

def get_dataloaders(params):
    
    train_dataloaders,test_dataloaders = loaders(params)

    return train_dataloaders,test_dataloaders

def get_datasets(params):
    
    train_dataset,test_dataset = loaders(params,get_dataset=True)

    return train_dataset,test_dataset

def loaders(params,get_dataset=False):
    
    if params.mean_norm == True:
        transform = transforms.Compose([transforms.Resize((params.img_sz,params.img_sz)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=params.mean_val, std=params.std_val)])
    else:
        transform = transforms.Compose([transforms.Resize((params.img_sz,params.img_sz)),transforms.ToTensor()])
    
    if params.pretext=='rotation':
        collate_func=rotnet_collate_fn
    else:
        collate_func=default_collate

    annotation_file = 'flowers_recognition_train.csv'                                 
    
    train_dataset = FlowersDataset(params,annotation_file,\
                            data_type='train',transform=transform)
    
    annotation_file = 'flowers_recognition_test.csv'                                  
    
    test_dataset = FlowersDataset(params,annotation_file,\
                                  data_type='test',transform=transform)

    if get_dataset:
        
        return train_dataset,test_dataset
    
    dataloader_train = DataLoader(train_dataset,batch_size=params.batch_size,\
                            collate_fn=collate_func,shuffle=True)
        
    dataloader_test = DataLoader(test_dataset,batch_size=params.batch_size,\
                            collate_fn=collate_func,shuffle=True)
    
    return dataloader_train,dataloader_test
#%%
if __name__=='__main__':
    config_file=r'D:\2020\Trainings\self-supervised-learning\config.yaml'
    cfg = utils.load_yaml(config_file,config_type='object')
    
#    tr_dset,ts_dset = get_datasets(cfg)

    tr_loaders,ts_loaders = get_dataloaders(cfg)
        
#    print ('length of tr_dset: {}'.format(len(tr_dset)))
#    print ('length of ts_dset: {}'.format(len(ts_dset)))

    
    data, label,idx,_ = next(iter(tr_loaders))
    print(data.shape, label) 

    data, label,idx,_ = next(iter(ts_loaders))
    print(data.shape, label)    

'''
This implementation is inspired from Aayush Agarwal Github
https://github.com/aayushmnit/Deep_learning_explorations/tree/master/8_Image_similarity_search

'''

import os,sys
import numpy as np
import argparse
import pickle
from tqdm import tqdm
from itertools import islice

import torch
#import torchvision.models as models
#from torch.utils.data import DataLoader 
#from torch.utils.data.dataloader import default_collate
#import torchvision.transforms as transforms

#from PIL import Image

#import matplotlib.pyplot as plt

from lshash.lshash import LSHash

import utils
import models

import dataloaders

#from flowers_dataloader import FlowersDataset,rotnet_collate_fn
#%%


def load_model(params,weight_path,device):
    # Load the model
    model = models.get_model(params)
    # Reload weights from the saved file
    print ('restoring weights from : ',weight_path)
    utils.load_checkpoint(os.path.join(weight_path), model,device)
    model.eval()
    return model

#def get_model():
#    resnet18 = models.resnet18(pretrained=True)
#    resnet18.eval()
#    return resnet18

# this is a hook (learned about it here: https://forums.fast.ai/t/how-to-find-similar-images-based-on-final-embedding-layer/16903/13)
# hooks are used for saving intermediate computations
class SaveFeatures():
    features=None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output): 
        output = output.view(output.size(0), -1)
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    def remove(self): 
        self.hook.remove()
        
def get_predictions(model,input_batch,device):
    input_batch = input_batch.to(device)
    with torch.no_grad():
        output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    #    print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        prob_output = torch.nn.functional.softmax(output, dim=0)
        class_output = np.argmax(prob_output,axis=1)
    
    return prob_output,class_output

def save_features_as_dict(model,dloader,sf,save_path,num_batch=10,device='cpu'):
    
    if num_batch!='all' and num_batch<len(dloader):
        dloader = islice(dloader,num_batch)
    else:
        num_batch = len(dloader)
        
    print('saving features for {} number of batches'.format(num_batch))
        
    img_names=()
        
    for input_batch,labels,idx,img_name in tqdm(dloader):
        prob,cls_opt = get_predictions(model,input_batch,device)
        img_names = img_names+img_name
        print (cls_opt)
    feature_dict = dict(zip(img_names,sf.features))
    ## Exporting as pickle
    pickle.dump(feature_dict, open(save_path, "wb"))
    return img_names,feature_dict
    
def save_embedding_hash(hash_params,save_path,img_names,features_dict):
    ## Locality Sensitive Hashing
    # params
    k = hash_params['hash_size'] # hash size
    L = hash_params['num_tables'] # number of tables
    d = hash_params['dim']# Dimension of Feature vector
    lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)
    
    # LSH on all the images
    for img_path, vec in tqdm(features_dict.items()):
        lsh.index(vec.flatten(), extra_data=img_path)
    ## Exporting as pickle
    pickle.dump(lsh, open(save_path, "wb"))


#%%

def main(cfg,checkpoint,pretrained_exp_dir,hash_params):
    
#    torch.manual_seed(cfg.seed)

    use_cuda = cfg.use_cuda and torch.cuda.is_available()
    cfg.use_cuda=use_cuda
    device = torch.device("cuda:{}".format(cfg.cuda_num) if use_cuda else "cpu")

    ## get the dataloaders
    dloader_train,_,dloader_test = dataloaders.get_dataloaders(cfg,val_split=None)
    
    
    # Load the model
    #params.num_classes=4
    model = load_model(cfg,os.path.join(pretrained_exp_dir,checkpoint),device)
    model = model.to(device)
    
    layer_name = model.avgpool
    sf = SaveFeatures(layer_name) ## Output before the last FC layer
    
    # define a folder for saving all embeddings
    embedding_path = os.path.join(pretrained_exp_dir,'embeddings')
    if not os.path.exists(embedding_path):
        os.makedirs(embedding_path)
    
    # save the feature embeddings for every image
    train_feat_path = os.path.join(embedding_path, 'train_features_dict.p')
    val_feat_path = os.path.join(embedding_path, 'val_features_dict.p')
    test_feat_path = os.path.join(embedding_path, 'test_features_dict.p')

#    img_names,features_dict = save_features_as_dict(model,dloader_train,sf,\
#                                            save_path=train_feat_path,num_batch='all',device)
##    img_names,features_dict = save_features_as_dict(model,dloader_val,sf,\
#                                            save_path=val_feat_path,num_batch='all',device)
    img_names,features_dict = save_features_as_dict(model,dloader_test,sf,\
                                            save_path=test_feat_path,num_batch='all',device=device)
    hash_path = os.path.join(embedding_path, 'features_hash.p')
    save_embedding_hash(hash_params,hash_path,img_names,features_dict)
    
if __name__=='__main__':
    #main()
    pretrained_exp_dir = r'D:\2020\Trainings\self_supervised_learning\experiments\self-supervised\ssl_exp_1'
    config_file='config_ssl.yaml'
    checkpoint = 'resnet18_4rot_epoch5_checkpoint.pth'
    yaml_path = os.path.join(pretrained_exp_dir,config_file)
    assert os.path.isfile(yaml_path), "No parameters config file found at {}".format(yaml_path)
    cfg = utils.load_yaml(yaml_path,config_type='object')
    cfg.root_path=r'D:\2020\Trainings\self_supervised_learning'

    hash_params={'hash_size':20,'num_tables':5,'dim':512}
    main(cfg,checkpoint,pretrained_exp_dir,hash_params)

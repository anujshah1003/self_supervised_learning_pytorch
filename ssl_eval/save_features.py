
'''
This SaveFeature function is taken from Aayush Agarwal Github
https://github.com/aayushmnit/Deep_learning_explorations/tree/master/8_Image_similarity_search

'''
import os,sys
sys.path.append(os.path.join(os.path.dirname("__file__"),'..'))

import numpy as np
import argparse
import pickle
from tqdm import tqdm
from itertools import islice

import torch
from torchsummary import summary

#import matplotlib.pyplot as plt
import utils
import models

import dataloaders
from utils.helpers import visualize

#from flowers_dataloader import FlowersDataset,rotnet_collate_fn

#from pretraining import dataloaders

#from pretraining.dataloaders.custom_dataset_loader import DatasetLoader,rotnet_collate_fn

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

torch.set_default_tensor_type(dtype)
#%%

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
        
def get_predictions(model,input_batch):
    input_batch = input_batch.to(device)
    with torch.no_grad():
        output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    #    print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        prob_output = torch.nn.functional.softmax(output, dim=0)
        class_output = torch.argmax(prob_output,axis=1)
    
    return prob_output,class_output

def save_features_as_dict(model,dloader,sf,feat_path,label_path,num_batch=10):
    
    if num_batch!='all' and num_batch<len(dloader):
        dloader = islice(dloader,num_batch)
    else:
        num_batch = len(dloader)
        
    print('saving features for {} number of batches'.format(num_batch))
        
    img_names=()
    labels=[]
        
    for input_batch,label,idx,img_name in tqdm(dloader):
        prob,cls_opt = get_predictions(model,input_batch)
        img_names = img_names+img_name
        labels = labels+label.tolist()
        #print (cls_opt)
    feature_dict = dict(zip(img_names,sf.features))
    labels_dict = dict(zip(img_names,labels))

    ## Exporting as pickle
    pickle.dump(feature_dict, open(feat_path, "wb"))
    pickle.dump(labels_dict, open(label_path, "wb"))
    return img_names,feature_dict

#%%
if __name__=='__main__':
    
    experiment_dir = r'D:\2020\Trainings\self_supervised_learning_pytorch\experiments\self_supervised\ssl_rotnet'
    restore_file = 'resnet18_best.pth'
    config_file = 'config_ssl.yaml'
    
    yaml_path = os.path.join(experiment_dir, config_file)
    assert os.path.isfile(yaml_path), "No parameters config file found at {}".format(yaml_path)
    cfg = utils.load_yaml(yaml_path,config_type='object')
#    params = utils.Params(yaml_path)
    
    use_cuda = cfg.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:{}".format(cfg.cuda_num) if use_cuda else "cpu")
    cfg.use_cuda=cfg.use_cuda
    cfg.batch_size=1
    cfg.root_path = r'D:\2020\Trainings\self_supervised_learning_pytorch'
    
    # for dataloader
    cfg.data_aug=False
    cfg.pretext=None
    cfg.num_classes=5
    
    # Load the model
    model = models.get_model(cfg)
    # Reload weights from the saved file
    print ('restoring weights from : ',restore_file)
    ckpt_path = os.path.join(experiment_dir,restore_file)
    utils.load_checkpoint(os.path.join(ckpt_path), model,device)
    model.eval()
    model.to(device)
    
    ## get the dataloaders
    torch.manual_seed(0)
    np.random.seed(0)
  
    cfg.val_split=None
    dloader_train,dloader_val,dloader_test = dataloaders.get_dataloaders(cfg)
    
#    layer_name = model.avgpool
#    layer_name=model.layer4[1]#model.layer2[1].bn2
#    layer_name=model.layer3[1]
    layer_name=model.layer2[1]
    summary(model,(3,128,128))

    
    # save the feature embeddings for every image
    features_path=os.path.join(experiment_dir,'features_3')
    if not os.path.exists(features_path):
        os.makedirs(features_path)
    
    # save the feature embeddings for every image
    train_feat_path = os.path.join(features_path, 'train_features_dict.p')
    train_label_path = os.path.join(features_path, 'train_labels_dict.p')
#    val_feat_path = os.path.join(features_path, 'val_features_dict.p')
#    val_label_path = os.path.join(features_path, 'val_labels_dict.p')
    test_feat_path = os.path.join(features_path, 'test_features_dict.p')
    test_label_path = os.path.join(features_path, 'test_labels_dict.p')
    
    sf = SaveFeatures(layer_name) ## Output before the last FC layer


    img_names,features_dict = save_features_as_dict(model,dloader_train,sf,\
                                            feat_path=train_feat_path,label_path=train_label_path,num_batch='all')
    print('sf features',sf.features.shape)
    
    sf = SaveFeatures(layer_name)
#    img_names,features_dict = save_features_as_dict(model,dloader_val,sf,\
#                                            feat_path=val_feat_path,label_path=val_label_path,num_batch='all')
    img_names,features_dict = save_features_as_dict(model,dloader_test,sf,\
                                            feat_path=test_feat_path,label_path=test_label_path,num_batch='all')
    print('sf features',sf.features.shape)
    

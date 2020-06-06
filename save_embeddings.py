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
import torchvision.models as models
from torch.utils.data import DataLoader 
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms

from PIL import Image

import matplotlib.pyplot as plt

from lshash.lshash import LSHash

import utils
import models

from flowers_dataloader import FlowersDataset,rotnet_collate_fn
#%%

def get_data(params):

    if params.mean_norm == True:
        transform = transforms.Compose([transforms.Resize((params.img_sz,params.img_sz)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=params.mean_pix, std=params.std_pix)])
    else:
        transform = transforms.Compose([transforms.Resize((params.img_sz,params.img_sz)),transforms.ToTensor()])

    if params.pretext =='rotation':
        collate_func=rotnet_collate_fn
    else:
        collate_func=default_collate
        

    annotation_file = 'train_labels.csv'                                 
    
    train_dataset = FlowersDataset(params,annotation_file,
                                  data_type='train',transform=transform)
    
    annotation_file = 'test_labels.csv'                                     
    
    test_dataset = FlowersDataset(params,annotation_file,\
                                  data_type='test',transform=transform)
    
    dataloader_train = DataLoader(train_dataset,batch_size=params.batch_size,\
                            collate_fn=collate_func,shuffle=True)
    
    dataloader_test = DataLoader(test_dataset,batch_size=params.batch_size,\
                            collate_fn=collate_func,shuffle=True)
    
    return dataloader_train,dataloader_test

def get_model(params,weight_path):
    # Load the model
    model = models.load_net(params)
    # Reload weights from the saved file
    print ('restoring weights from : ',weight_path)
    utils.load_checkpoint(os.path.join(weight_path), model)
    model.eval()
    return model

def get_model():
    resnet18 = models.resnet18(pretrained=True)
    resnet18.eval()
    return resnet18

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
        class_output = np.argmax(prob_output,axis=1)
    
    return prob_output,class_output

def save_features_as_dict(model,dloader,sf,save_path,num_batch=10):
    
    if num_batch!='all' and num_batch<len(dloader):
        dloader = islice(dloader,num_batch)
    else:
        num_batch = len(dloader)
        
    print('saving features for {} number of batches'.format(num_batch))
        
    img_names=()
        
    for input_batch,labels,idx,img_name in tqdm(dloader):
        prob,cls_opt = get_predictions(model,input_batch)
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

def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tire',
                        help='Dataset name (default: CIFAR10)')
    parser.add_argument('--root_path', default=r'D:\2020\project_small_data\Tire_inspection\tire_inspection_cropped_data_final', help="Directory containing the dataset")

    parser.add_argument('--experiment_path', type=str, default='exp_1',
                        help='the name of the experiment (dir where all the \
                        log files and trained weights of the experimnet will be saved)')

    parser.add_argument('--restore_file', default='rotNet_tire_resnet-18_4rot_epoch0_lr_checkpoint.pth', help="name of the file in --experiment_name \
                     containing weights to load")
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    
    args.experiment_path = r'D:\2020\project_small_data\Small_Data\pretraining\experiment_dir\exp_1'
    args.restore_file = 'rotNet_tire_resnet-18_lr_best.pth'
    
    yaml_path = os.path.join(args.experiment_path, 'params.yaml')
    assert os.path.isfile(yaml_path), "No parameters config file found at {}".format(yaml_path)
    params = utils.Params(yaml_path)
    
    use_cuda = params.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:{}".format(params.cuda_num) if use_cuda else "cpu")
    params.use_cuda = use_cuda
    ## get the dataloaders
    params.root_path=args.root_path
    params.pretraining = None
    dloader_train,dloader_val,dloader_test = get_data(params)
    
    # Load the model
    params.num_classes=4
    model = get_model(params,os.path.join(args.experiment_path, args.restore_file))
    model = model.to(device)
    
    layer_name = model.avg_pool
    sf = SaveFeatures(layer_name) ## Output before the last FC layer
    
    # save the feature embeddings for every image
    train_feat_path = os.path.join(args.experiment_path, 'train_features_dict.p')
    val_feat_path = os.path.join(args.experiment_path, 'val_features_dict.p')
    test_feat_path = os.path.join(args.experiment_path, 'test_features_dict.p')

    img_names,features_dict = save_features_as_dict(model,dloader_train,sf,\
                                            save_path=train_feat_path,num_batch='all')
    img_names,features_dict = save_features_as_dict(model,dloader_val,sf,\
                                            save_path=val_feat_path,num_batch='all')
    img_names,features_dict = save_features_as_dict(model,dloader_test,sf,\
                                            save_path=test_feat_path,num_batch='all')
    hash_params={'hash_size':20,'num_tables':5,'dim':18432}
    hash_path = os.path.join(args.experiment_path, 'features_hash.p')
    save_embedding_hash(hash_params,hash_path,img_names,features_dict)
    
if __name__=='__main__':
    main()

'''
Data loader for loading data for self-supervised learning for rotation task
'''
import os
from PIL import Image
import utils
from utils.transformations import rotate_img,visualize
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import pandas as pd
from itertools import islice

#%%
class FlowersDataset(Dataset):
    """ Flowers Dataset Class loader """
    
    def __init__(self,params, annotation_file,data_type='train', \
                 transform=None):
        
        """
        Args:
            image_dir (string):  directory with images
            annotation_file (string):  csv/txt file which has the 
                                        dataset labels
            transforms: The trasforms to apply to images
        """
        
        self.data_path = os.path.join(params.root_path,params.data_path,params.imgs_dir)
        self.label_path = os.path.join(params.root_path,params.data_path,params.labels_dir,annotation_file)
        self.transform=transform
        self.pretext = params.pretext
        if self.pretext == 'rotation':
            self.num_rot = params.num_rot
        self._load_data()

    def _load_data(self):
        '''
        function to load the data in the format of [[img_name_1,label_1],
        [img_name_2,label_2],.....[img_name_n,label_n]]
        '''
        self.labels = pd.read_csv(self.label_path)
        
        self.loaded_data = []
#        self.read_data=[]
        for i in range(self.labels.shape[0]):
            img_name = os.path.join(self.data_path, self.labels['FileName'][i])
            #data.append(io.imread(os.path.join(self.image_dir, self.labels['img_name'][i])))
            label = self.labels['Label'][i]
            img = Image.open(img_name)
            self.loaded_data.append((img,label,img_name))
            img.load()#This closes the image object or else you will get too many open file error
#            self.read_data.append((img,label))

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):

        idx = idx % len(self.loaded_data)
        img,label,img_name = self.loaded_data[idx]
#        img = io.imread(img_name)
#        img = Image.open(img_name)   
        img,label = self._read_data(img,label)
        
        return img,label,idx,img_name

    def _read_data(self,img,label):
        
        '''
        function to read the data
        '''
        if self.pretext == 'rotation':
        # if in rotnet mode define a loader function that given the
        # index of an image it returns the 4 rotated copies of the image
        # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
        # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
            if self.num_rot == 4:
                rotated_imgs = [
                    self.transform(img),
                    self.transform(rotate_img(img, 90).copy()),
                    self.transform(rotate_img(img, 180).copy()),
                    self.transform(rotate_img(img, 270).copy())
                ]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                
            elif self.num_rot == 3:
                rotated_imgs = [
                    self.transform(img),
                    self.transform(rotate_img(img, 120).copy()),
                    self.transform(rotate_img(img, 240).copy())
                ]
                rotation_labels = torch.LongTensor([0,1,2])
        
            return torch.stack(rotated_imgs, dim=0), rotation_labels
                        
        else:
            # supervised mode; if in supervised mode define a loader function 
            #that given theindex of an image it returns the image and its 
            #categorical label
            img = self.transform(img)
            return img, label
        
def rotnet_collate_fn(batch):
    
    batch = default_collate(batch)
    #assert(len(batch) == 2)
    batch_size, rotations, channels, height, width = batch[0].size()
    batch[0] = batch[0].view([batch_size * rotations, channels, height, width])
    batch[1] = batch[1].view([batch_size * rotations])
    return batch
#%%
if __name__ == '__main__':
     
    config_path = r'D:\2020\Trainings\self-supervised-learning\config\config.yaml'
    cfg = utils.load_yaml(config_path,config_type='object')
    if cfg.mean_norm == True:
        transform = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=cfg.mean_val, std=cfg.std_val)])
    else:
        transform = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),transforms.ToTensor()])
    
    if cfg.pretext=='rotation':
        collate_func=rotnet_collate_fn
    else:
        collate_func=default_collate

    annotation_file = 'flowers_recognition_train.csv'                                 
    
    dataset = FlowersDataset(cfg,annotation_file,\
                            data_type='test',transform=transform)
    
    len(dataset)
    img,label,idx,img_name = dataset[1]
    
    visualize(img,label)
    
    data_val = DataLoader(dataset,batch_size=10,collate_fn=collate_func,shuffle=False)
    
    data, label,idx,img_names = next(iter(data_val))
    print(data.shape, label)
        
    for data,label,idx,img_names in islice(data_val,4):
        print(data.shape, label)
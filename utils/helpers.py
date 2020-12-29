import yaml
import os
import torch
import shutil
import logging
import matplotlib.pyplot as plt
import numpy as np

class dotdict(dict):
    """dot.notation access to dictionary attributes
    This function is taken from stackoverflow answer by farrell
    and derek- https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_yaml(config_file,config_type='dict'):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        #params = yaml.load(f,Loader=yaml.FullLoader)
        
    if config_type=='object':
        cfg = dotdict(cfg)
    return cfg
    
def save_yaml(config,save_path='config.yaml'):
    if type(config)!=dict:
        config=dict(config)
    with open(save_path, 'w') as file:
        yaml.dump(config, file)
        
def save_checkpoint(state, is_best, save_path, checkpoint='checkpoint.pth', best_model='model_best.pth'):
    """ Save model. """
    os.makedirs(save_path, exist_ok=True)
    torch.save(state, save_path + '/' + checkpoint)
    if is_best:
        shutil.copyfile(save_path + '/' + checkpoint, save_path + '/' + best_model)

def load_checkpoint(checkpoint, model, device='cuda',optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """

    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint,map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'],strict=False)
    except:
        model.load_state_dict(checkpoint,strict=False)

#    state = torch.load(path, map_location='cuda:0')
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

#    return checkpoint
        
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)
        
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
        
def visualize(input_arr,labels):
    
    fig = plt.figure(figsize=(12,12))
    if len(input_arr.shape)==4:
        num_imgs=input_arr.shape[0]
    else:
        num_imgs=1
        input_arr=np.expand_dims(input_arr,axis=0)
        labels=[labels]
    rows=np.sqrt(num_imgs)
    rows=int(np.ceil(rows))
    for i in range(num_imgs):
      plt.subplot(rows,rows,i+1)
      plt.tight_layout()
#      img = np.rollaxis(img,0,3)
      img = input_arr[i]
      img = np.rollaxis(img,0,3)
      plt.imshow(img, interpolation='none')
      plt.title("class_label: {}".format(labels[i]))
      plt.xticks([])
      plt.yticks([])
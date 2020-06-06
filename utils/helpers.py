import yaml
import os
import torch
import shutil

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
    with open(save_path, 'w') as file:
        yaml.dump(config, file)
        
def save_checkpoint(state, is_best, save_path, checkpoint='checkpoint.pth', best_model='model_best.pth'):
    """ Save model. """
    os.makedirs(save_path, exist_ok=True)
    torch.save(state, save_path + '/' + checkpoint)
    if is_best:
        shutil.copyfile(save_path + '/' + checkpoint, save_path + '/' + best_model)

def load_checkpoint(checkpoint, model, optimizer=None,cuda_num=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if cuda_num==None:
        map_location='cpu'
    else:
        map_location='cuda:{}'.format(cuda_num)
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint,map_location=map_location)
    try:
        model.load_state_dict(checkpoint['state_dict'],strict=False)
    except:
        model.load_state_dict(checkpoint,strict=False)

#    state = torch.load(path, map_location='cuda:0')
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

#    return checkpoint
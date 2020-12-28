'''
This implementation is inspired from Aayush Agarwal Github
https://github.com/aayushmnit/Deep_learning_explorations/tree/master/8_Image_similarity_search

'''


import os
import pickle

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import argparse

def get_similar_item(idx, feature_dict, lsh_variable, n_items=5):
    response = lsh_variable.query(feature_dict[list(feature_dict.keys())[idx]].flatten(), 
                     num_results=n_items+1, distance_func='hamming')
    # each response is a tuple where the first element has the feature vector 
    #and image name and the second element is the distance from queried image
    columns = 3
    rows = int(np.ceil(n_items+1/columns))
    fig=plt.figure(figsize=(2*rows, 3*rows))
    for i in range(1, columns*rows +1):
        if i<n_items+2:
            img = Image.open(response[i-1][0][1])
            name='dist:{}'.format(np.round(response[i-1][1],2))
            fig.add_subplot(rows, columns, i,title=name)
            fig.subplots_adjust(bottom=0.1, right=1, top=1.2)
            plt.imshow(img)
    return plt.show()

if __name__=='__main__':
    
    #main()
    pretrained_exp_dir = r'D:\2020\Trainings\self_supervised_learning\experiments\self-supervised\ssl_exp_1'
    feat_file='test_features_dict.p'
    hash_file='features_hash.p'
    num_sim_imgs=5
    
    feat_path = os.path.join(pretrained_exp_dir,'embeddings',feat_file)
    hash_path = os.path.join(pretrained_exp_dir,'embeddings',hash_file)

    assert os.path.isfile(feat_path), "No features dictionary found at {}".format(feat_path)
    assert os.path.isfile(hash_path), "No features hash found at {}".format(hash_path)

    feature_dict = pickle.load(open(feat_path,'rb'))

    lsh = pickle.load(open(hash_path,'rb'))
    
    qry_img_idx = 100

    get_similar_item(qry_img_idx,feature_dict, lsh,num_sim_imgs)


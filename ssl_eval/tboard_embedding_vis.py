import os,cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
#import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
#from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard.plugins import projector

tf.__version__


# define the log dor for saving checkpoints
experiment_dir = r'D:\2020\Trainings\self_supervised_learning\experiments\self-supervised\ssl_rotnet'

LOG_DIR = 'embedding_logs_3'
embedding_path = os.path.join(experiment_dir,LOG_DIR)

if not os.path.exists(embedding_path):
    os.makedirs(embedding_path)
    
#%%
#load features
features_path=os.path.join(experiment_dir,'features_3')
train_features=os.path.join(features_path,'train_features_dict.p')
train_labels=os.path.join(features_path,'train_labels_dict.p')

with open(train_features, 'rb') as f:
    features_dict = pickle.load(f)
with open(train_labels, 'rb') as f:
    labels_dict = pickle.load(f)
#feature_vectors = np.loadtxt('feature_vectors_400_samples.txt')
feature_vectors=[]
img_names=[]
labels = []

for k,v in features_dict.items():
    img_names.append(k)
    feature_vectors.append(features_dict[k])
    labels.append(labels_dict[k])

feature_vectors=np.asarray(feature_vectors) 

print ("feature_vectors_shape:",feature_vectors.shape)
print ("num of images:",feature_vectors.shape[0])
print ("size of individual feature vector:",feature_vectors.shape[1])

#%%
# prepare meta data file
labels_to_class={0:'daisy',1:'dandelion',2:'rose',3:'sunflower',4: 'tulip'}
metadata_file = open(os.path.join(embedding_path, 'metadata.tsv'), 'w')
metadata_file.write('Class\tName\n')

for label in labels:
    name=labels_to_class[label]
    print(name,label)
    metadata_file.write('{}\t{}\n'.format(label,name))

metadata_file.close()

#%%   
#prepare sprite images         
#root_path = r'D:\2020\project_small_data\Tire_inspection\tire_inspection_cropped_data_final'
#data_dir='train'
img_data=[]
for img_name in tqdm(img_names):
    input_img=cv2.imread(img_name)
    input_img_resize=cv2.resize(input_img,(32,32))
    img_data.append(input_img_resize)
    
img_data = np.array(img_data)

#%%
# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data
#%%
sprite = images_to_sprite(img_data)
cv2.imwrite(os.path.join(embedding_path, 'sprite.png'), sprite)
#scipy.misc.imsave(os.path.join(LOG_DIR, 'sprite.png'), sprite)

#%%
# if features size is huge you can use PCA to reduce it

from sklearn.decomposition import PCA,KernelPCA
pca = PCA()
kpca = KernelPCA()

x_pca = pca.fit_transform(feature_vectors)
#x_pca = pd.DataFrame(x_pca)
#x_pca.head()

explained_variance = pca.explained_variance_ratio_
explained_variance

feat_kpca = kpca.fit_transform(feature_vectors)

features = tf.Variable(feat_kpca, name='features')
# Create a checkpoint from embedding, the filename and key are
# name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=features)
checkpoint.save(os.path.join(embedding_path, "embedding.ckpt"))

# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    # Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path =  'metadata.tsv'
    # Comment out if you don't want sprites
embedding.sprite.image_path =  'sprite.png'
embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
    # Saves a config file that TensorBoard will read during startup.

projector.visualize_embeddings(embedding_path, config)
#%%

'''

# Use the iris dataset to illustrate PCA:
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
df.head()

from sklearn.preprocessing import StandardScaler
variables = ['sepal length','sepal width','petal length','petal width']
x = df.loc[:, variables].values
y = df.loc[:,['target']].values
x = StandardScaler().fit_transform(x)
x = pd.DataFrame(x)

from sklearn.decomposition import PCA,KernelPCA
pca = PCA()
kpca = KernelPCA()

x_pca = pca.fit_transform(x)
x_pca = pd.DataFrame(x_pca)
x_pca.head()

explained_variance = pca.explained_variance_ratio_
explained_variance

feat_kpca = kpca.fit_transform(feature_vectors)

x_pca['target']=y
x_pca.columns = ['PC1','PC2','PC3','PC4','target']
x_pca.head()

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1') 
ax.set_ylabel('Principal Component 2') 
ax.set_title('2 component PCA') 
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
 indicesToKeep = x_pca['target'] == target
 ax.scatter(x_pca.loc[indicesToKeep, 'PC1']
 , x_pca.loc[indicesToKeep, 'PC2']
 , c = color
 , s = 50)
ax.legend(targets)
ax.grid()

#for the transformed data, get the explained variances
expl_var_pca = np.var(x_pca, axis=0)
expl_var_kpca = np.var(x_kpca, axis=0)
print('explained variance pca: ', expl_var_pca)
print('explained variance kpca: ', expl_var_kpca)

expl_var_ratio_pca = expl_var_pca / np.sum(expl_var_pca)
expl_var_ratio_kpca = expl_var_kpca / np.sum(expl_var_kpca)

print('explained variance ratio pca: ', expl_var_ratio_pca)
print('explained variance ratio kpca: ', expl_var_ratio_kpca)





#https://towardsdatascience.com/dimension-reduction-techniques-with-python-f36ca7009e5c
'''
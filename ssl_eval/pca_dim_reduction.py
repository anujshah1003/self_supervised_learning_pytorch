import os
import numpy as np
import pickle
from sklearn.decomposition import PCA,KernelPCA

experiment_dir = r'D:\2020\Trainings\self_supervised_learning_pytorch\experiments\self_supervised\ssl_rotnet'

feature_path=os.path.join(experiment_dir,'features_1')
train_feat_path = os.path.join(feature_path,'train_features_dict.p')
test_feat_path = os.path.join(feature_path,'test_features_dict.p')
train_label_path = os.path.join(feature_path,'train_labels_dict.p')
test_label_path = os.path.join(feature_path,'test_labels_dict.p')

assert os.path.isfile(train_feat_path), "No features dictionary found at {}".format(train_feat_path)
assert os.path.isfile(test_feat_path), "No features dictionary found at {}".format(test_feat_path)

assert os.path.isfile(train_label_path), "No labels dictionary found at {}".format(train_label_path)
assert os.path.isfile(test_label_path), "No labels dictionary found at {}".format(test_label_path)
    
train_feature_dict = pickle.load(open(train_feat_path,'rb'))
test_feature_dict = pickle.load(open(test_feat_path,'rb'))

train_label_dict = pickle.load(open(train_label_path,'rb'))
test_label_dict = pickle.load(open(test_label_path,'rb'))

labels=[]
for img in train_feature_dict.keys():
    label=train_label_dict[img]
    labels.append(int(label))

x_tr = np.array(list(train_feature_dict.values())) 
y_tr = np.array(labels)

print(x_tr.shape)
#x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.18, random_state=42)


pca = PCA()
#kpca = KernelPCA()

x_tr_pca = pca.fit_transform(x_tr)
#x_pca = pd.DataFrame(x_pca)
#x_pca.head()

explained_variance = pca.explained_variance_ratio_
explained_variance
np.sum(explained_variance[0:300])

#feat_kpca = kpca.fit_transform(feature_vectors)
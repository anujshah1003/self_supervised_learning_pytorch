# self_supervised_learning

## Introduction
Supervised-Learning although used ubiqitously and shown its syregth in various industrial application have a major caveta that it requires annotated data.

Self-Supervision(type of unsupervised learning) on the other hand is emerging in this context by not requring any expensive labeing and it does so by automatically generating labels form the data itself.

Self-supervised Learning can be used to pretrain the network ad then fine-tune on the main task, in the scenario where less labelled data is available. 
### Drawbacks of a system that requires umpteen labelled data
    Labeling data is often expensive.
    Experts are required to provide labels to the data
    A common scenario - few labeled data are available due to the constrain of budget or the experimentation is expensive.
    An intelligent system should learn from few labeled samples (As Humans do)

When you incoporate self-supervision in your training pipeline, two terms you ought to know is
#### Pretext Task - The proxy task used for self supervised training (e.g. rotation)
#### Downstream Task - The actual Task in your hand

The idea then is to pre-train networks via pretext tasks that do not require expensive manual annotations and can be automatically generated from the data itself. Once pre-trained, networks can be applied to a target task by using only a modest amount of labelled data.

## Self-Supervision (Pretext Task)
The dataset and problem I am using is flowers category recognition form kaggle - https://www.kaggle.com/alxmamaev/flowers-recognition
As we have discussed, self-supervision is abut defining a pretext task for which labels are generated form the data itself. One such simplest pretext task is Rotation. The idea is very simple - take an image X and rotate it by various degree, for e.g. (0,90,180). so you input is image and its corresponding label is the degree of orientation. And for here on you train it in the traditional supervised way. Once learned on rotation this network can bes used for transfer learning for your mainn task.

#### Pretext Task (predicting degree of rotation)
![alt text](https://github.com/anujshah1003/self_supervised_learning/blob/master/readme_imgs/rotation.png)


#### Downstream Task (main task of predicting flower categories)
![alt text](https://github.com/anujshah1003/self_supervised_learning/blob/master/readme_imgs/main_task.png)

## Implementing Rotation based Self-supervised Learning
We will implement this paper - ![UNSUPERVISED REPRESENTATION LEARNING BY PREDICTING IMAGE ROTATIONS ](https://arxiv.org/abs/1803.07728)

Offical Github - https://github.com/gidariss/FeatureLearningRotNet

My video on the paper - https://www.youtube.com/watch?v=R-9V7MV9GPw&t=751s
## Steps to follow
Step-1: Split the data intro train and test (check the eda notebook to see the data).

Step-2: Take out 10% of the training data as a small labeled data to evaluate the efficiency of self-supervised learning.

Step-3: train on this small labeled data and compute the results on test data

Step-4: Do self-supervised learning on entire training data without using their labels

Step-5: use the pretrained rotnet model and fine tune on the small labeled data

Step-6: use the above finetuned model and evaluate the perfromance on test data.

Step-7: Compare the test results from step-3 with step-6 to see the benefits of unsupervised/self-supervised pretraining

## Perfromance boost with Rotnet pretraining
![alt text](https://github.com/anujshah1003/self_supervised_learning_pytorch/blob/master/readme_imgs/rotnet_tboard.png)

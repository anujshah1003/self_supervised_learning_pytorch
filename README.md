# self_supervised_learning

## Introduction

Supervised-Learning although used ubiqitously and shown its syregth in various industrial application have a major caveta that it requires annotated data.

Self-Supervision(type of unsupervised learning) on the other hand is emerging in this context by not requring any expensive labeing and it does so by automatically generating labels form the data itself.

Self-supervised Learning can be used to pretrain the network ad then fine-tune on the main task, in the scenario where less labelled data is available. 
### Motivation for learning from small data
    Labeling data is often expensive.
    Experts are required to provide labels to the data
    A common scenario - few labeled data are available due to the constrain of budget or the experimentation is expensive.
    An intelligent system should learn from few labeled samples (As Humans do)

When you incoporate self-supervision in your training pipeline, two terms you ought to know is
#### Pretext Task - The proxy task used for self supervised training (e.g. rotation)
#### Downstream Task - The actual Task in your hand

The idea then is to pre-train networks via pretext tasks that do not require expensive manual annotations and can be automatically generated from the data itself. Once pre-trained, networks can be applied to a target task by using only a modest amount of labelled data.

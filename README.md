# README

Logion is a system for infilling text and detecting corruptions in ancient Greek works. 
Here we document the base model which one can use for inference (suggestion generation for infilling unknown gaps of text). 
We provide the base model, along with several specialized models which we explain below, on Dropbox (as they are too large to store here).
We also provide code (the "beam search") designed to perform inference on gaps of more than one token, along with the training code which one can use to fine-tune the models, provided they have the recommended hardware specifications.

While beam search and stochastic gradient descent have been well-explored in the literature, corruption detection is a significantly more complicated task, and to our knowledge, the algorithm we implement is explained neither in existing literature nor in the body of the paper.
For this reason, we do not include the code here, but can provide it upon request.

## System requirements

It is recommended, but not required, that your system has a GPU in order to perform inference with Logion. On a system with Python >=3.8.8;
Conda >=4.10.1, one can execute<br/>
```
>> conda create --name logion pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
>> conda activate logion
```
to initialize the environment. 

### System recommendations for training
Logion was trained on a research computing cluster with 2.8 GHz Intel Ice Lake nodes for several days. If you intend to fine-tune, it's recommended that your processor has at least 128 GB of memory and a GPU. With a Nvidia K80 / T4 (standard on Google Colab), beam search should take no more than 10 seconds for spans of up to 10 tokens, with current specifications.

## Model

Stored on [dropbox](https://www.dropbox.com/sh/x8lsd6la7meq4xk/AABb8tqHPTT1KHYvLvKLeaEta?dl=0). 
In addition to the base model, trained according to the standard objective for masked-language modeling, we provide fine-tunings which specialize
in predicting spans of more than one consecutive missing token. For each 1<k<=5, we train a "expert<i>k</i>span" model which is never trained on missing spans of fewer than k missing tokens.
At inference time, we use the expert models to predict appropriate spans, defaulting to the expert5span model for gaps of more than 5 missing tokens (such gaps are statistically rare in the corpus).

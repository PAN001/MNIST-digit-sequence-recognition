# Introduction

There is a readme.html which is converted from this markdown file for demonstration.

`CTCLoss.py`

- Pytorch implementation of Connectionist Temporal Classification (CTC) loss function by extending `torch.autograd`

`CNN_LSTM_FC_model.py`

- A Pytoch based CNN+LSTM+CTC model

`train.py`

- Trainig script

# Code

For the sake of convenience, a test shell script test.sh is provided for testing the pretrained model on test set. Simply run the following, and it will help set up the environment and fit the model automatically:

```shell
sh test.sh
```

## Environment Set Up
The running environment is set up in Python 2.7.10.
By running `virtualenv`, it could help set up the environment based on the `requirements.txt` easily:

```shell
# Create and activate new virtual environment
virtualenv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Code Description

# Connectionist Temporal Classification (CTC)

## Forward & Backward

# Decoding

Given the probability distribution $P(l|x)$, we can compute a label $l$ for an input sequence $x$ by taking the most likely label. Thus, given that $L^{\leqslant T}$ is the set of sequences of length less than or equal to $T$ with letters drawn from the alphabet $L$, we can express our desired classifier h(x) as follows:

$$ h(x) = \arg \max_{\ell \in L^{\le T}} P(\ell | x) $$

Computing the most likely $l$ from the probability distribution $P(l|x)$ is known as decoding. However, given that the alphabet $L$ and the maximum sequence length $T$ may be quite large, it is computationally intractable to examing every possible $l \in L^{\leqslant T}$. There is no known algorithm to efficiently compute this $h(x)$ precisely; however, there are several ways to approximate decoding which work well enough in practice.

Traditionally, decoding is done in one of following two ways.

## Best Path Decoding

The first traditional decoding strategy is best path decoding, which assumes that the most likely path corresponds to the most likely label. This is not necessarily true: suppose we have one path with probability $0.1$ corresponding to label A, and ten paths with probability $0.05$ each corresponding to label B. Clearly, label B is preferable overall, since it has an overall probability of $0.5$; however, best path decoding would select label A, which has a higher probability than any path for label B.

Best path decoding is fairly simple to compute; simply look at the most active output at every timestep, concatenate them, and convert them to a label (via removing blanks and duplicates). Since at each step we choose the most active output, the resulting path is the most likely one.

# Network Architecture

Basically, the images are first processed by a CNN to extract features, then these extracted features are fed into a RNN. Then, the output from RNN are fed into a softmax layer to convert each output to a probability distribution over 11 classes (i.e. 10 digits and 1 blank). Finally, the probability distribution is the input to the final CTC layer.

The architecture of CNN is just Convolution + Batch Normalization + Relu activation + Max Pooling for simplicity and specifically LSTM is used as RNN units.

## Model-1: sCNN (CNN with samll kernei size) + LSTM + CTC

The first model has the following architecture:

|           | Model-1: sCNN (CNN with samll kernei size) + LSTM + CTC       |
|-----------|---------------------------------------------------------------|
| Conv1     | 1 input channel, 5*5 kernel size, 10 feature map, stride = 1  |
| Maxpool1  | 10 input channel, 2*2 kernel size, stride = 1                 |
| Conv2     | 10 input channel, 5*5 kernel size, 20 feature map, stride = 1 |
| Maxpool2  | 20 input channel, 2*2 kernel size, stride = 1                 |
| Batchnorm |                                                               |
| Dropout   | p = 0.5                                                       |
| LSTM      | 32 hidden size, 1 hidden layer                                |
| Softmax   | =>11                                                          |
| CTC       |                                                               |


### Experiment Result
- training(20)/test(5)

    Validation set: Average loss: 0.4893, Average edit dist: 0.1348

- training(20)/test(20)

    Validation set: Average loss: 1.6601, Average edit dist: 0.4023

- training(20)/test(100)

    Validation set: Average loss: 16.9200, Average edit dist: 4.4844

## Model-2: lCNN (CNN with large kernei size) + LSTM + CTC

The second model uses a kernel size with the same height as the image. This is required by the write-up.

|           	| Model-1: lCNN (CNN with large kernei size) + LSTM + CTC       	|
|-----------	|---------------------------------------------------------------	|
| Conv1     	| 1 input channel, 36*2 kernel size, 10 feature map, stride = 1 	|
| Maxpool1  	| 10 input channel, 1*2 kernel size, stride = 1                 	|
| Conv2     	| 10 input channel, 1*2 kernel size, 20 feature map, stride = 1 	|
| Maxpool2  	| 20 input channel, 1*2 kernel size, stride = 1                 	|
| Batchnorm 	|                                                               	|
| LSTM      	| 32 hidden size, 1 hidden layer                                	|
| Softmax   	| =>11                                                          	|
| CTC       	|                                                               	|

## Model-3: lCNN (CNN with large kernei size) + BLSTM + CTC

The third model uses a kernel size with the same height as the image. Rather than the typical LSTM, it also uses bidirectional LSTM. 

|           	| Model-1: lCNN (CNN with large kernei size) + LSTM + CTC       	|
|-----------	|---------------------------------------------------------------	|
| Conv1     	| 1 input channel, 36*2 kernel size, 10 feature map, stride = 1 	|
| Maxpool1  	| 10 input channel, 1*2 kernel size, stride = 1                 	|
| Conv2     	| 10 input channel, 1*2 kernel size, 20 feature map, stride = 1 	|
| Maxpool2  	| 20 input channel, 1*2 kernel size, stride = 1                 	|
| Batchnorm 	|                                                               	|
| LSTM      	| 32 hidden size, 1 hidden layer                                	|
| Softmax   	| =>11                                                          	|
| CTC       	|                                                               	|
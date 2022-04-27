# Training a network for homography prediction

A homography is a transformation relating two projections of a planar surface.
It is a 3x3 matrix, H, that maps an image coordinates of a point on a plane to
coordinates in another image. Since we are using homogeneous corrdinates, the
scale of H doesn't matter, so a solutions requires finding (at least) four
matching points. The homography can then be solved for using linear least
squares.

Traditional CV methods for doing this involve key point detection (e.g. SIFT)
followed by correspondence matching (e.g. min distance + RANSAC).

Here we try to replace these steps by training a DNN. We use the model
HomographyNet (described in https://arxiv.org/pdf/1606.03798.pdf) as the
starting point with which to experiment.

To train our network we have a synthetic COCO-based dataset described as
follows:
- The two channels of the input represent two projections of the same planar
  scene from different viewpoints, in grayscale.
- The target is a vector containing offsets of four corners of a random crop of
  the first channel, giving their positions in the second channel. The four
  corner offsets can be used to generate the eight equations to solve the
  homography, and can thus be said to encode the homography.

## Setup
1. Create a virtual environment and install the required dependencies as
follows:
```bash
# Clone this repo
cd homography-dl
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

2. Download training and test data to your local filesystem from:
https://drive.google.com/drive/folders/1ikm8MuB34-38xNS5v1dzZOBUJLXUV4ch

Now test that you can the complete training and testing loop on a toy amount of
data:
```bash
python3 run.py \
        --test-data=<path_to/test.h5> \
        --train-data=<path_to/train.h5> \
        --mini=True \
        --normalize=False
```

## Running on a GPU remotely
Note: this can be run on a GPU by using
[colab.research.google.com](colab.research.google.com) as follows:
1. Select `Edit > Notebook settings > Hardware accelerator > GPU`.
2. In a code cell, clone this repo and `cd` into the directory:
```
!git clone <repo url>
!cd homography-dl
```
3. Log into Google drive, and locate the above data directory in your 'Shared
with me' folder. Add a shortcut to your 'My Drive' folder.
4. Add a code cell to mount this folder, and copy it locally:
```
from google.colab import drive
drive.mount('/content/drive')
!mkdir /content/data
!cp /content/drive/MyDrive/homography/test.h5 /content/data/
!cp /content/drive/MyDrive/homography/train.h5 /content/data/
!ls /content/data/
```
5. Now you can execute `python3 run.py` to train on a GPU via the
`--device=cuda` option. Test to see this is working by adding a code cell with:
```bash
!python3 run.py \
         --test-data=/content/data/test.h5 \
         --train-data=/content/data/train.h5 \
         --device=cuda \
         --mini=True \
         --normalize=False
```

## Model architechture and design decisions
HomographyNet uses a VGG-like Network with eight convolutional layers (the CNN
backbone that does the key point detection work of SIFT) and two fully connected
layers (which predict the corner offsets). The key points of this archtecture
are:
- Conv layers and Relu non-linearity as the basic feature detection block.
- Pooling layers, so convolutions layers detect features over a range of scales
  (with more complex features further down the network, so more channels per
  layer).
- Batch normalization to improve stability for a given learning rate.
- Dropout after the fully connected layers for regularization (not the
  convolutional layers, as (1) they contain far fewer parameters, so require
  less regularization, and (2) have less of a regularizing effect anyway when
  applied to highly correlated signals like images/feature maps which are fed
  into subsequent convolutional layers).
See `class Model` in `run.py` for the Pytorch implementation.

For the training (and interleaved validation) phase, we use mean squared error
(MSE) loss. This is essentially the same metric as mean average corner error
(MACE), which we use for evaluating the trained model, but less computationally
expensive.

For the optimizer, I achieved the best result using AdamW with default Pytorch
hyperparameters. In the results section I describe the how this performs in
terms of overfitting. Note that using the optimizer described in the
HomographyNet paper (SGD with lr=0.005, momentum=0.9) I saw exploding gradients.
Adaptive optimizers (like AdamW) are tolerant to a range of learning rates
relative to standard SGD with momentum, so are a good 'first try' given the
limited time I was willing to spend tuning hyperparameters.

## Results
The following command was used to train and test the network on a machine with
a GPU.
```bash
python3 run.py \
        --test-data=<path_to/test.h5> \
        --train-data=<path_to/train.h5> \
        --device=cuda \
        --batch-size=64 \
        --epochs=12
```
Training with a batch size of 64 is the largest power-of-2 batch size that can
fit on the single GPU used (Tesla K80, 11.4GB, according to
`torch.cuda.get_device_properties(0)`). This gave a throughput of slightly over
160 images per second while training, resulting in around 10 minutes per epoch.

This above comnand trains the network for 12 epochs, generating the right of the
three loss curves below.

![Loss Curves](images/loss-curves.png)

The model state that gave the best validation loss (in this case the final
state) is loaded, and the MACE is calculated on the test set. **A final MACE
of 8.14 is achieved** with the above command.

This is similar to - in fact slightly better than - that reported in the
HomographyNet paper (9.20).

If I had more time I would have trained for many more epochs in this
configuration.

The other two graphs show training performance for two configurations when
running with `--dropout=False`. The first demonstrates the overfitting you
get without the regularization that you get from dropout. In theory, L2
regularization via weight decay should also prevent the fitting. However, the
middle figure shows how a weight decay of 0.001 with AdamW performs poorly in
comparison to dropout.

## Limitations, improvements and further work

1. Something I'm sceptical of (and not commented on in the HomographyNet paper)
is the ability for a model trained on a synthetic data set generated in this
way to generalize well on real world data. This is because a homography only
relates two projections in the scenarios of:
- Rotation only movements
- Planar scenes
- Scenes in which objects are very far from the viewer.
But in our synthetic dataset, none of these assumptions are guaranteed to hold.
The below image pairs taken from the test set are an example when the
assumptions are broken.

![Dog Head](images/dog-head.png)

Traditional CV homography estimation techniques (e.g. SIFT + RANSAC) do not suffer from this problem, as they do not require large synthetic data sets to train.

2. The HomographyNet paper (June, 2016) is now nearly 6 years old, which is a long
time in the CV/ML world. We can see
[here](https://paperswithcode.com/sota/homography-estimation-on-pds-coco) that
HomographyNet was surpassed as the SOTA architechture for homography estimation
in 2019 by PFNet (and again with a more recent iteration). With more time, I'd
like to reimplement this model in Pytorch. `run.py` could be easily extended to
support more models via a command-line argument.
(Note that paperswithcode.com reports HomographyNet as achieving a MACE of 2.5.
I'm not sure why this is, as it doesn't agree with what is reported in the
paper).

PFNet is a much deeper network more FLOPs per iteration) than HomographyNet,
but has a similar number of parameters (as >90% of HomographyNet's parameters
are in its penultimate FC layer) so we can expect a similar number of epochs,
but greater time to train the network.

### TODO

- Follow-up work:
    - understand (by means of visualization) how learned features of first layer differ for homography estimation networks vs image classifiers (as first conv layer for imagenet ResNet, for example, learns filters for RGB layers, whereas homography estimators trained on COCO learn filters for 2 separate grayscale perspective projections)

- Comment on the limitations of how well this would generalize to a real-world test set, vs 

    - In the generated dataset we are using, none of these assumptions are guaranteed to hold. So a model trained on this data may lead to bad generalization on real world test data.
    - Traditional CV homography estimation techniques (e.g. SIFT + RANSAC) do not suffer from this problem, as they do not require large synthetic data sets to train.


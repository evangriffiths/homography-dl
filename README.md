# Training a network for homography prediction

Abstract TODO

## Background
TODO

## Setup
1. Create a virtual environment and install the required dependencies as
follows:
```bash
git clone ...
cd ...
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

2. Download training and test data to your local filesystem from:
https://drive.google.com/drive/folders/1ikm8MuB34-38xNS5v1dzZOBUJLXUV4ch

Now test you can the complete training and testing loop on a toy amount of data:
```bash
python3 run.py --
```


## Running on GPU remotely
Note: this can be run on a GPU by using colab.research.google.com as follows:
1. Select `Edit > Notebook settings > Hardware accelerator > GPU`.
2. Copy and paste run.py into a code cell:
```
%%writefile run.py
<run.py copied here>
```
3. Log into Google drive, and locate the above data directory in your 'Shared with me' folder. Add a shortcut to your 'My Drive' folder.
4. Add a code cell to mount this folder, and copy it locally:
```
from google.colab import drive
drive.mount('/content/drive')
!mkdir /content/data
!cp /content/drive/MyDrive/homography/test.h5 /content/data/
!cp /content/drive/MyDrive/homography/train.h5 /content/data/
!ls /content/data/
```
5. Now you can execute `python3 run.py` to train on a GPU via the `--device=cuda` option. Test to see this is working:
```
python3 run.py --test-data=/content/data/test.h5 --train-data=/content/data/train.h5 --device=cuda --mini=True
```

## Results
The following command was can be used to train and test the network on a
machine with a GPU.
```
$ python3 run.py --test-data=<path_to/test.h5> --train-data=<path_to/train.h5> --device=cuda --batch-size=64 --epochs=9
```
Training with a batch size of 64 is the largest power-of-2 batch size that can
fit on the single GPU used (spec unknown). This gave a throughput of slightly
over 160 images per second while training, resulting in around 10 minutes per
epoch.

This above comnand trains the network for 9 epochs, generating the validation
MSE loss curve below TODO.
As you can see, the model converges after 6 epochs. The model state after the
6th epoch is loaded, and the MACE is calculated on the test set. A final MACE
of 9.55 is achieved.

This is similar, though slightly worse to that reported in the HomographyNet
paper (9.20).

## Improvements and further work
TODO
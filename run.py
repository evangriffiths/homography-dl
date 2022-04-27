import argparse
import copy
import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tt
from tqdm import tqdm

def get_mean_and_var(file_path):
    """
    Get mean and variance of a HDF5 dataset. Assumes input shape of
    [channels, width, height], and returns means and variances of shape
    [channels].
    """
    data_info = h5py.File(file_path, 'r')
    summed, summed_squared = 0, 0
    for idx in range(len(data_info)):
        data = data_info[str(idx)]
        image = torch.tensor(np.array(data), dtype=torch.float32)
        # Mean over height, width dimensions
        summed += torch.mean(image, dim=[1, 2])
        summed_squared += torch.mean(image**2, dim=[1, 2])

    mean = summed / len(data_info)
    var = (summed_squared / len(data_info)) - mean**2
    return mean, var


def get_data_loader(dataset, args):
    """
    Return a torch dataloader object for iterating over a dataset.
    """
    if (args.mini is True):
        samples = int(len(dataset) / 1000)
        sampler = torch.utils.data.RandomSampler(dataset, num_samples=samples,
                                                 replacement=True)
    else:
        sampler=None

    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                       sampler=sampler)


class HDF5Dataset(torch.utils.data.Dataset):
    """
    Subclass the the torch dataset class so we can use the torch dataloader
    for easy batching, etc. of the dataset.
    Inspired by towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5.
    """
    def __init__(self, file_path, transform=None):
        super().__init__()
        self.data_info = h5py.File(file_path, 'r')
        self.transform = transform

    def __len__(self):
        return len(self.data_info['/'])

    def __getitem__(self, index):
        data = self.data_info[str(index)]
        image = torch.tensor(np.array(data), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(data.attrs.get('label').flatten(),
                             dtype=torch.float32)
        return (image, label)


class Model(nn.Module):
    """
    Implementation of 'HomographyNet' (see https://arxiv.org/pdf/1606.03798.pdf
    Fig. 1.)
    Code nearly identical to implementation from
    github.com/mazenmel/Deep-homography-estimation-Pytorch, with dropout layers
    added, as described in the paper.
    """
    def __init__(self, dropout):
        super(Model,self).__init__()
        self.dropout = dropout
        self.layer1 = nn.Sequential(nn.Conv2d(2, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())        
        self.layer6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.dropout = nn.Dropout(p=0.5)
        # Output of final conv layer produces 128 16x16 feature maps
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        # Final linear layer predicts the eight values defining the four
        # corner offsets:
        # [tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y]
        # where:
        # tl = top left corner offset
        # tr = top right corner offset
        # br = bottom right corner offset
        # bl = bottom left corner offset
        self.fc2 = nn.Linear(1024, 8)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        if self.dropout is True:
            x = self.dropout(x)
        x = x.view(-1, 128 * 16 * 16)
        x = self.fc1(x)
        if self.dropout is True:
            x = self.dropout(x)
        x = self.fc2(x)
        return x


def mean_average_corner_error(predicted, label):
    """
    A loss criterion for evaluating a predicted homography by comparing
    predicted and ground truth (i.e. label) corner locations.
    """
    diff = label.subtract(predicted) # (B, 8) error per corner dimension
    diff = diff.pow(2) # (B, 8) squared error per corner dimension
    diff = diff.reshape((-1, 4, 2)) # (B, 4, 2) squared error by corner
    diff = diff.sum(-1) # (B, 4) sum the squared error per corner
    diff = diff.sqrt() # (B, 4) L2 distance per corner
    diff = diff.mean(-1) # (B,) mean of L2 distance per sample
    mace = diff.mean(-1) # (1,) mean over all of test dataset
    return mace


def test(data_loader, model, criterion, device):
    """
    A generic function for testing a model using some data, based on some
    criterion.
    """
    model.eval()
    running_loss = 0
    data_loader_ = tqdm(data_loader, desc="\tTesting:", total=len(data_loader),
                        ncols=70)
    for image, label in data_loader_:
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, label)
        running_loss += loss

    mean_loss = running_loss / len(data_loader)
    return mean_loss.to('cpu').item()


def train_val(train_loader, val_loader, model, criterion, optimizer, num_epochs,
              device):
    """
    A generic function for simultaneously training and validate a model over
    some number of epochs using some data, based on some criterion.
    """
    best_params = None
    best_loss = None
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))

        # Train
        model.train()
        running_train_loss = 0
        train_loader_ = tqdm(train_loader,
                             desc="\tTraining:",
                             total=len(train_loader),
                             ncols=70)
        for image, label in train_loader_:
            optimizer.zero_grad()
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = criterion(output, label)
            running_train_loss += loss
            loss.backward()
            optimizer.step()

        mean_train_loss = running_train_loss / len(train_loader)

        # Validate
        val_loss = test(data_loader=val_loader,
                         model=model,
                         criterion=criterion,
                         device=device)

        print("\tMean train loss:", mean_train_loss.to('cpu').item())
        print("\tMean validation loss:", val_loss)

        # Save model parameters correspodning to best validation result
        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            best_params = copy.deepcopy(model.state_dict())

    # Return the model with parameters that produced the lowest validation loss
    print("Saving model params with best validation loss: {}".format(best_loss))
    model.load_state_dict(best_params)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Learning the homography between two images via deep '
                    'learning on the COCO dataset.')
    parser.add_argument('--test-data', type=str, required=True,
                        help='The path to the hdf5 file containing test data')
    parser.add_argument('--train-data', type=str, required=True,
                        help='The path to the hdf5 file containing training '
                             'data')
    parser.add_argument('--batch-size', type=int, default=1, required=False,
                        help='The batch size')
    parser.add_argument('--epochs', type=int, default=1, required=False,
                        help='The number of epochs to train over')
    parser.add_argument('--device', type=str, default='cpu', required=False,
                        help='The device the model is executed on',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--normalize', type=bool, default=False, required=False,
                        help='If true, normalize the image data that is the '
                             'input to the model')
    parser.add_argument('--dropout', type=bool, default=True, required=False,
                        help='If true, include dropout layers after the fully '
                             'connected layers')
    parser.add_argument('--mini', type=bool, default=False, required=False,
                        help='If true, use a toy subsample of the dataset')
    args = parser.parse_args()

    torch.manual_seed(0)

    device = torch.device(args.device)
    model = Model(args.dropout).to(device)

    # Training and validation
    print("Training and validation phase:")
    if args.normalize is True:
        train_val_mean, train_val_var = get_mean_and_var(args.train_data)
    transform = tt.Normalize(train_val_mean,
                             train_val_var**0.5) if args.normalize else None
    train_val_data_set = HDF5Dataset(args.train_data, transform=transform)
    training_length = int(len(train_val_data_set) * 0.8)
    train_data_set, val_data_set = torch.utils.data.random_split(
        dataset=train_val_data_set,
        lengths=[training_length, len(train_val_data_set) - training_length])
    train_loader = get_data_loader(train_data_set, args)
    val_loader = get_data_loader(val_data_set, args)
    final_model = train_val(train_loader=train_loader,
                            val_loader=val_loader,
                            model=model,
                            criterion=torch.nn.MSELoss(),
                            optimizer=torch.optim.AdamW(model.parameters()),
                            num_epochs=args.epochs,
                            device=device)

    # Testing
    print("Final evalutaion on test data:")
    if args.normalize is True:
        test_mean, test_var = get_mean_and_var(args.test_data)
    test_transform = tt.Normalize(test_mean,
                                  test_var**0.5) if args.normalize else None
    test_data_set = HDF5Dataset(args.test_data, transform=test_transform)
    mean_mace = test(data_loader=get_data_loader(test_data_set, args),
                     model=final_model,
                     criterion=mean_average_corner_error,
                     device=device)
    print("Final mean MACE:", mean_mace)

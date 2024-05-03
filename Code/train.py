import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize as param
from torch.linalg import multi_dot as md
import os
import pickle
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
from torch._utils import _accumulate
from tqdm import trange, tqdm
import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt

from getData import *
from utils import *
from Model import *

def train(task:str, model, optimizer, criterion, train_loader, test_loader, max_epoch:int, decay_eff:float, decay_epochs:int, device, output_data:str, output_model:str, min_epoch=0,
          test_accs=[], train_losses=[], epochs_list=[], SA1_max=[], SA1_min=[], SA2_max=[], SA2_min=[], SB_max=[], SB_min=[], SC_max=[], SC_min=[]):
    ''' Main loop for training LurieNet on sCIFAR-10, sMNIST, psMNIST datasets.
        args:
            task: string declaring the task to be trained on. Accepts "scifar10", "smnist", "psmnist".
            model: instance of the LurieNet to be trained.
            optimizer: the chosen optimizer.
            criterion: chosen loss function.
            train_loader: data loader with training samples.
            test_loader: data loader with test samples.
            max_epoch: number of epochs.
            decay_eff: the scalar to multiply the learning rate by at the specified epochs.
            decay_epochs: list containing the epochs to perform a learning rate cut on.
            device: hardware which the code is being run on.
            output_data: path to the logged data.
            output_model: path to the stored models.
optional args:
            min_epoch: epoch to begin training from (default 0). Useful for resuming training after interruption.
            test_accs: list to store test accuracy after each epoch (default empty). Useful for passing in previously recorded list before interruption.
         train_losses: list to store train losses after each epoch (default empty). Useful for passing in previously recorded list before interruption.
          epochs_list: list of epochs which have already passed (counter starts from 1). Useful for passing in previously recorded list before interruption.
              SA1_max: list of max eigenvalues in SA1 over time.
              SA1_min: list of min eigenvalues in SA1 over time.
              SA2_max: list of max eigenvalues in SA2 over time.
              SA2_min: list of min eigenvalues in SA2 over time.
              SB_max: list of max eigenvalues in SB over time.
              SB_min: list of min eigenvalues in SB over time.
              SC_max: list of max eigenvalues in SC over time.
              SC_min: list of min eigenvalues in SC over time.
     returns:
            model: optimized model.
            optimizer: optimizer with up to date parameter choices.
            test_accs: history of test accuracies for each epoch.
            train_losses: history of mean (across batches) loss for each epoch.
    '''

    if task != "scifar10" and task != "smnist" and task != "psmnist":
        raise ValueError('This task cannot be run!')

    optim_params = list(model.parameters())

    for epoch in trange(min_epoch, max_epoch): # trange displays progress meter of loops

        model.train()
        loss_epoch = []

        for i, (inputs, targets) in enumerate(train_loader):

            inputs, targets = inputs.to(device), targets.to(device) # moves data to gpu if available

            # CIFAR-10 images are rgb = 3 and 32x32 = 1024 pixels. MNIST images are rgb = 1 and 28x28 = 784 pixels.
            # Input will be of shape [batch size, rgb, width, height].
            # Want rgb to be the last variable so we can present image pixel by pixel.
            if task == "scifar10":
                inputs = inputs.view(-1, 3, int(1024)).permute(0,2,1) # [batch size, pixels, rgb] = [batch size, 1024, 3]
            else:
                inputs = inputs.view(-1, int(784)).unsqueeze(dim = 2) # [batch size, pixels, rgb] = [batch size, 784, 1]

            # forward + backward + optimize
            outputs = model(inputs) # outputs has shape [batch size, output_size]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_epoch.append(loss.item())

        # track loss over time, and have it reflect the mean loss over an epoch so it is more reflective of training trends than last batch loss.
        mean_loss = np.mean(loss_epoch)
        train_losses.append(mean_loss)

        # track eigen/singular values over time
        if model.A == None:
          SA1_max.append(torch.min(torch.abs(model.SA1)).to('cpu').detach().numpy()) # smallest mag of SA1 corresponds to SA1_max when passed through parametrization
          SA1_min.append(torch.max(torch.abs(model.SA1)).to('cpu').detach().numpy()) # largest mag of SA1 corresponds to SA1_min when passed through parametrization
          SA2_max.append(torch.min(torch.abs(model.SA2)).to('cpu').detach().numpy()) # smallest mag of SA2 corresponds to SA2_max when passed through parametrization
          SA2_min.append(torch.max(torch.abs(model.SA2)).to('cpu').detach().numpy()) # largest mag of SA2 corresponds to SA2_min when passed through parametrization
        if model.B == None:
          SB_max.append(torch.min(torch.abs(model.SB)).to('cpu').detach().numpy()) # smallest mag of SB corresponds to SB_max when passed through parametrization
          SB_min.append(torch.max(torch.abs(model.SB)).to('cpu').detach().numpy()) # largest mag of SB corresponds to SB_min when passed through parametrization
        if model.C == None:
          SC_max.append(torch.min(torch.abs(model.SC)).to('cpu').detach().numpy()) # smallest mag of SC corresponds to SC_max when passed through parametrization
          SC_min.append(torch.max(torch.abs(model.SC)).to('cpu').detach().numpy()) # largest mag of SC corresponds to SC_min when passed through parametrization

        # Note: when logging data epoch 1 represents LurieNet after first pass over the data and epoch zero represents initial model.
        epoch_log = epoch + 1
        print('Epoch {}, mean batch loss {}'.format(epoch_log, mean_loss))

        # Update learning rate when specified.
        optimizer = lr_scheduler(epoch, optimizer, decay_eff, decay_epochs)

        # Testing model.
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for (inputs, targets) in test_loader:
                inputs, targets = inputs.to(device), targets.to(device) # moves data to gpu if available.

                if task == "scifar10":
                    inputs = inputs.view(-1, 3, int(1024)).permute(0,2,1)
                else:
                    inputs = inputs.view(-1, int(784)).unsqueeze(dim = 2)

                outputs = model(inputs)

                _, predicted = torch.max(outputs, 1) # predicted: index of the class with the highest value for each image in batch.
                total += targets.shape[0] # updates total with the number images in the batch.
                correct += (predicted == targets).sum().item() # updates correct with the number of correct predictions from the batch.

            print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))
            test_accs.append((100.0 * float(correct) / float(total))) # Used floats to get exact accuracy.

        epochs_list.append(epoch_log)

        # save model for every epoch as storage space required is quite small, can have training disruptions with colab
        model_path = os.path.join(output_model,"epoch" + str(epoch_log) + ".pt")
        torch.save(model.state_dict(), model_path)

        # save stats so far, will overwrite every time as rows are added to dataframe.
        stats_path = os.path.join(output_data, "stats.csv")
        cur_stats = pd.DataFrame()
        cur_stats["epoch"] = epochs_list
        cur_stats["loss"] = train_losses # loss over time: indicating mean loss over the batches so it is more reflective of training trends than last batch loss.
        cur_stats["test_acc"] = test_accs # test acc after each epoch.

        # eigen/singular values over time.
        if model.A == None:
          cur_stats["SA1_max"] = SA1_max
          cur_stats["SA1_min"] = SA1_min
          cur_stats["SA2_max"] = SA2_max
          cur_stats["SA2_min"] = SA2_min
        if model.B == None:
          cur_stats["SB_max"] = SB_max
          cur_stats["SB_min"] = SB_min
        if model.C == None:
          cur_stats["SC_max"] = SC_max
          cur_stats["SC_min"] = SC_min

        cur_stats.to_csv(stats_path, index=False)

    return model, optimizer, test_accs, train_losses

def main():
    ### Hardware agnostic settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print('Default tensor type is now cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Device in use is: ", device)

    # set random seed to reproduce the work - ensures data is loaded in the same way, but initialisations are still random (not seeded).
    # Training is deterministic after model instantiated, so results should be reproducible using r_seed and loading initialisation of network.
    r_seed=int(torch.empty((), dtype=torch.int32).random_().item())
    torch.manual_seed(r_seed)

    # task related settings
    task = "scifar10" # will accept scifar10, psmnist, or smnist

    if task == "scifar10":
        input_size = 3 # rgb
        data_dir = "/Users/carlrichardson/Documents/Python/LurieNet/Data/CIFAR10/"
    else:
        input_size = 1 # black and white
        data_dir = "/Users/carlrichardson/Documents/Python/LurieNet/Data/"
    output_size = 10 # same for all tasks

    # data
    train_loader, val_loader, test_loader = getData(task, data_dir, device, r_seed)

    # Model settings
    n=256 # dimension of state
    m=256 # dimension of input to nonlinearity
    A=torch.eye(n,n) # can optionally pass A matrix as a fixed parameter
    B=torch.eye(n,m) # can optionally pass B matrix as a fixed parameter
    C=torch.eye(m,n) # can optionally pass C matrix as a fixed parameter
    eta = 0.01 # upper limit on gradient
    delta=0.002 # discrete time step of ode
    k=1 # dimension of k-compound system
    g=1 # slope restriction of nonlinearity
    gb=2*(k**2)*eta/(3*delta) # max. sum of singular values for B matrix
    gc=k # max. sum of singular values for C matrix
    ga1=2*(k**2)*eta/(3*delta) # bounds eigenvalues of symmetric component of A
    ga2=2*(k**2)*eta/(3*delta) # bounds terms in skew-symmetric component of the special orthogonal transformation
    v1_ind=True # indicates if input v1 is used by model
    v2_ind=False # indicates if input v2 is used by model
    init = 'He' # specified weight initialisation. Accepts 'He', 'Xavier', 'RNNsofRNNs', 'He+RNNsofRNNs'
    model = LurieNet(input_size, output_size, n, m, k, g, gb, gc, ga1, ga2, delta,
                    v1_ind, v2_ind, init, C=C).to(device) # move model to gpu if available

    # Training settings.
    # Same as RNNs of RNNs paper, however different settings are used after hyperparameter optimization
    num_epochs = 100
    lr = 0.001
    lr_scale_epochs = [90] # epochs to perform a learning rate cut
    lr_scalar = 0.1 # scalar to multiply the learning rate by at the specified epochs
    weight_decay = 1e-5

    # Optimizer
    optim_params = list(model.parameters())
    optimizer = torch.optim.Adam(optim_params, lr=lr, weight_decay=weight_decay)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Datalogging
    model_name="RNNsofRNNs"
    hyperparam_settings = "k%d_lr%.3f_delta%.3f_1ga%.3f_2ga%.3f_gb%.3f_gc%.3f" % (k, lr, delta, ga1, ga2, gb, gc)
    model_dir= os.path.join(".../LurieNet/Models/"+model_name+"/"+task,hyperparam_settings)
    log_dir= os.path.join(".../LurieNet/Experiments/"+model_name+"/"+task,hyperparam_settings)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    else:
        raise ValueError(f"Do not overwrite existing data!")
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    else:
        raise ValueError(f"Do not overwrite existing data!")

    # save initialization of network
    torch.save(model.state_dict(), os.path.join(model_dir,"epoch0.pt"))

    # dictionaries for logging all information about experiment and hyperparameters
    exp_setup = {'r_seed': r_seed, 'device': device,  'model_name': model_name, 'model_dir': model_dir, \
    'log_dir': log_dir, 'task':task}
    model_hyperparameters = {'n':n, 'm':m, 'eta':eta, 'k':k, 'g':g, 'gb':gb, 'gc':gc, 'ga1':ga1, 'ga2':ga2, 'delta':delta, \
                    'v1_ind':v1_ind, 'v2_ind':v2_ind, 'init':init, 'A':A, 'B':B, 'C':C}
    train_hyperparameters = {'num_epochs':num_epochs, 'lr':lr, 'lr_scale_epochs':lr_scale_epochs, \
                            'lr_scalar':lr_scalar, 'weight_decay':weight_decay, 'optimizer':optimizer, \
                                'criterion': criterion}

    # logging experiments and hyperparameters
    with open(log_dir+'/exp_setup.pkl', 'wb') as f:
        pickle.dump(exp_setup, f)
        f.close()
    with open(log_dir+'/model_hyperparameters.pkl', 'wb') as f:
        pickle.dump(model_hyperparameters, f)
        f.close()
    with open(log_dir+'/train_hyperparameters.pkl', 'wb') as f:
        pickle.dump(train_hyperparameters, f)
        f.close()

    print('Total params. being trained: %.3fk \n' % (sum(p.numel() for p in model.parameters())/1000.0))
    if A!=None or B!=None or C!=None:
        print("Have all hyperparameters been correctly set? Be particularly careful when passing in A, B or C!\n")
    else:
        print("Have all hyperparameters been correctly set? \n")

    # train
    model, optimizer, test_accs, train_losses = train(task, model, optimizer, criterion, train_loader, test_loader, num_epochs,
                                                        lr_scalar, lr_scale_epochs, device, log_dir, model_dir)
    
    return 0
    
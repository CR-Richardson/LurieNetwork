import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt



def get_data(data_path:str, batch:int, bs:int, test_split:float, rseed:int, device):
    """
    Prepare training data.
    args:
            data_path: Path to data.
                batch: Number of batches to split full dataset into.
                   bs: Batch size (number of trajectories per batch).
           test_split: ratio of total batches to be used as test data.
                rseed: Random seed.
                device: hardware used.
    returns:
              X_train: (train_batch, bs, tmax, n) training dataset.
               X_test: (test_batch, bs, tmax, n) test dataset.
                    T: Array of continuous time points.
            exp_setup: Settings used for generating data.   
    """

    with open(data_path + 'exp_setup.npy', 'rb') as f:
        exp_setup = np.load(f, allow_pickle=True)
    with open(data_path + 'X.npy', 'rb') as f:
         X = torch.from_numpy( np.load(f) ) # (tmax, N=batch*bs, n)
    with open(data_path + 'T.npy', 'rb') as f:
         T = torch.from_numpy( np.load(f) )

    exp_setup = exp_setup.tolist()
    t0, tf, step = exp_setup['time']
    tmax, N, n = X.shape

    X = torch.transpose(X,0,1) # (N, tmax, n)
    X = X.view(batch, bs, tmax, n) # (batch, bs, tmax, n)
    
    # split data into train and test sets
    rng = np.random.default_rng(seed=rseed+1)
    test_batch = int(batch*test_split)
    test_batch = rng.choice( np.arange(0, batch), size=test_batch, replace=False )
    train_batch = np.delete( np.arange(0, batch), test_batch )    
    X_test = X[test_batch]
    X_train = X[train_batch]
    
    return X_train.double(), X_test.double(), T.double(), exp_setup



def lr_scheduler(epoch:int, optimizer, decay:float, decay_epochs:list):
    """
    Decay learning rate by a factor decay for every epoch in decay_epochs.
    args:
               epoch: Current epoch of training loop.
           optimizer: Optimizer with parameters from previous epoch.
               decay: Scalar to multiply lr by.
        decay_epochs: List containing the epochs which the lr should be cut at.
    returns:
           optimizer: Same optimizer as before with updated lr.
    """

    if epoch in decay_epochs:
      for param_group in optimizer.param_groups:
          param_group['lr'] = decay*param_group['lr']

      print( 'New learning rate is: %.4f' % ( param_group['lr']) )

    return optimizer



def train(X:torch.tensor, X2:torch.tensor, T:torch.tensor, model, optimizer, criterion, 
          max_epoch:int, model_dir:str, decay:float, decay_epochs:list, model_name:str, device, min_epoch=0, train_losses=[], test_losses=[]):
    """
    Training loop.
    args:
                X: Training data with shape (batch_train, bs, dtmax, n).
               X2: Test data with shape (batch_test, bs, tmax, n).
                T: Time tensor.
            model: Instantiation of the chosen model.
        optimizer: Chosen optimizer.
        criterion: Chosen loss function.
        max_epoch: Epoch which training will terminate at.
        model_dir: Path to where models and data are stored.
            decay: Scalar to multiply lr by.
     decay_epochs: List containing the epochs which the lr should be cut at.
       model_name: Name of model in use.
           device:
        min_epoch: Epoch to begin training at.
    returns:
            model: Final version of the model.
            stats: Dataframe containing training and test loss.
    """

    # train data
    batch, bs, tmax, n = X.shape
    # X0 = X[:,:,0,:]

    # test data
    batch2, _, _, _ = X2.shape
    # X20 = X2[:,:,0,:]

    if len(train_losses) == 0:
        train_losses = []

    if len(test_losses) == 0:
        test_losses = []
    
    for epoch in range(min_epoch, max_epoch):
        
        # training loop
        model.train()
        batch_losses = []

        for i in range(batch):
            optimizer.zero_grad()
            
            if model_name == 'NeuralODE/':
                input1 = X[i,:,0,:].to(device)
                input2 = T.to(device)
                target = X[i].to(device)
                output = model(input2, input1) # (bs, tmax, n)
                
            else:
                input1 = X[i,:,0,:].to(device)
                target = X[i].to(device)
                output = model(input1) # (bs, tmax, n)
                
            loss = criterion(target, output)
            loss.backward()
            optimizer.step()
            batch_losses.append( loss.item() )
            # check_kcon(model)

        mean_train_loss = np.mean(batch_losses) # loss over epoch
        train_losses.append(mean_train_loss)

        optimizer = lr_scheduler(epoch, optimizer, decay, decay_epochs)
        
        # testing loop
        model.eval()
        batch_losses = []
        
        with torch.no_grad():
            for i in range(batch2):
                
                if model_name == 'NeuralODE/':
                    input1 = X2[i,:,0,:].to(device)
                    input2 = T.to(device)
                    target = X2[i].to(device)
                    output = model(input2, input1) # (bs, tmax, n)
                    del input2
                else:
                    input1 = X2[i,:,0,:].to(device)
                    target = X2[i].to(device)
                    output = model(input1) # (bs, tmax, n)
                
                loss = criterion(target, output)
                batch_losses.append( loss.item() )                

        mean_test_loss = np.mean(batch_losses) # mean loss over epoch
        test_losses.append(mean_test_loss)

        print( 'Epoch %d/%d - Train loss: %.3f - Test loss: %.3f' % (epoch + 1, max_epoch, mean_train_loss, mean_test_loss) )

        # save models
        model_path = model_dir + 'epoch{:03d}.pt'.format(epoch+1)
        torch.save(model.state_dict(), model_path)
            
        # save stats.
        stats_path = model_dir + 'stats.csv'
        stats = pd.DataFrame()
        stats["train loss"] = train_losses # mean loss over epoch
        stats["test loss"] = test_losses # mean loss over epoch
        stats.to_csv(stats_path, index=False)

        del input1
        del target
        torch.cuda.empty_cache()
        
    return model, stats



def check_kcon(model):
    """
    Function for verifying k-contraction condition is met.
    """

    n, m = model.B.shape
    A = model.A
    B = model.B
    C = model.C
    k = model.k
    g = model.g


    # calculate alpha_k
    A_sym = A + A.t()
    A_sym_eigs, _ = torch.sort( torch.linalg.eigvals(A_sym).real, dim=0, descending=True )
    alpha_k = torch.sum( A_sym_eigs[:k] ) / 2*k

    # calculate LHS of inequality
    B_sing, _ = torch.sort( torch.linalg.svdvals(B), dim=0, descending=True )
    C_sing, _ = torch.sort( torch.linalg.svdvals(C), dim=0, descending=True )
    B_sing2 = B_sing[:k] * B_sing[:k]
    C_sing2 = C_sing[:k] * C_sing[:k]
    LHS = g * g * torch.sum( B_sing2 * C_sing2 )

    # calculate RHS of inequality
    RHS = alpha_k * alpha_k * k

    if  (alpha_k > 0) or (LHS > RHS):
        print("k contraction conditions are not met!")
    
    return 0



def batch_preds(model, X, T, model_name:str):
    """
    Return predictions for all inputs.
    args:
             model: Instantiation of the chosen model.
                 X: Test data with shape (batch, bs, tmax, n).
                 T: Time tensor.
        model_name: Name of model.
    returns:
        preds: Outputs of model with shape (batch, bs, tmax, n).
    """

    model.eval()
    with torch.no_grad():
        batch, bs, tmax, n = X.shape
        preds = torch.zeros( (batch, bs, tmax, n) )
        X0 = X[:,:,0,:]
        
        for i in range(batch):

            if model_name == 'NeuralODE/':
                output = model(T, X0[i])
            else:
                output = model(X0[i])
                
            preds[i] = output

    return preds.detach()



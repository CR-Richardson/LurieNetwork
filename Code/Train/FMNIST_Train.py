import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import sys
import os
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



root = ''
sys.path.append(root + 'Code/')
sys.path.append(root + 'Code/Training')
sys.path.append(root + 'Code/Models')
import CNN
import kLurieNet_nonHurwitz as LN_k
import Decoder as dc



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



def train(train_loader, test_loader, max_epoch:int, model, dec, optimizer, criterion,
          decay:float, decay_epochs:list, model_dir:str, device, enc=0, stop_freeze=0, min_epoch=0,
          mean_train_losses=[], mean_test_losses=[], mean_train_acc=[], mean_test_acc=[]
          ):
    """
    Training loop.
    args:
          train_loader: Dataloader for training set. Each batch has shape = (bs, 1, n, n).
           test_loader: Dataloader for test set. Each batch has shape = (bs, 1, n, n).
             max_epoch: Epoch which training will terminate at.
                 model: Latent recurrent model.
                   dec: Decoder.
             optimizer: Chosen optimizer.
             criterion: loss function.
                 decay: Scalar to multiply lr by.
          decay_epochs: List containing the epochs which the lr should be cut at.
             model_dir: Path to where models and data are stored.
                device: Hardware in use.
    optional:
                    enc: Encoder.
           stope_freeze: (int) epoch where end-to-end training begins and the encoder no longer trains in isolation.
              min_epoch: Epoch which training will start at.
      mean_train_losses: List containing training loss at each epoch from disrupted run.
       mean_test_losses: List containing test loss at each epoch from disrupted run.
         mean_train_acc: List containing training accuracy at each epoch from disrupted run.
          mean_test_acc: List containing test accuracy at each epoch from disrupted run.
    returns:
      model, dec, stats: Final version of the model components and stats.
    """

    if len(mean_train_losses) == 0:
        mean_train_losses = []
        mean_train_acc = []
        mean_test_losses = []
        mean_test_acc = []


    for epoch in range(min_epoch, max_epoch):

        # training loop
        if enc != 0:
            enc.train()

        if epoch < stop_freeze:
            model.eval()
            dec.eval()
            print('epoch {0}: model, decoder in eval mode!'.format(epoch) )
        else:
            model.train()
            dec.train()
            print('epoch {0}: model, decoder in train mode!'.format(epoch) )

        batch_losses = []
        batch_acc = []

        for batch, data in enumerate(train_loader):

            inputs, labels = data
            inputs = inputs.double().to(device) # shape = (bs, 1, d, d)
            labels = labels.to(device) # shape = (bs)

            optimizer.zero_grad()
            bs, _, d, _ = inputs.shape # d = horizontal pixels in image.

            if enc == 0:
                inputs = inputs.flatten(start_dim=2, end_dim=3).squeeze(1).to(device) # shape = (bs, d * d)
                print('Train inputs flattened!')

            ##### model outputs #####
            if enc != 0:
              inputs = enc(inputs) # (bs, X)
            latent = model(inputs) # (bs, X)
            output = dec(latent) # (bs, n_classes)
            pred = torch.argmax( F.softmax(output, dim=1), dim=1 ) # (bs)

            ##### compute loss #####
            loss = criterion(output, labels)

            ##### compute classification accuracy #####
            acc = torch.sum( pred == labels) / bs

            ##### optimise #####
            loss.backward()
            optimizer.step()

            batch_losses.append( loss.item() )
            batch_acc.append( acc.item() )

        mean_train_losses.append( np.mean(batch_losses) )
        mean_train_acc.append( np.mean(batch_acc) )

        optimizer = lr_scheduler(epoch, optimizer, decay, decay_epochs)

        # testing loop
        if enc != 0:
              enc.eval()
        model.eval()
        dec.eval()

        batch_losses = []
        batch_acc = []

        with torch.no_grad():
          for batch, data in enumerate(test_loader):

              inputs, labels = data
              inputs = inputs.double().to(device) # shape = (bs, 1, d, d)
              labels = labels.to(device) # shape = (bs)

              bs, _, d, _ = inputs.shape # d = horizontal pixels in image.
              
              if enc == 0:
                inputs = inputs.flatten(start_dim=2, end_dim=3).squeeze(1).to(device) # shape = (bs, d * d)
                print('Test inputs flattened!')

              ##### model outputs #####
              if enc != 0:
                inputs = enc(inputs) # (bs, X)
              latent = model(inputs) # (bs, X)
              output = dec(latent) # (bs, n_classes)
              pred = torch.argmax( F.softmax(output, dim=1), dim=1 ) # (bs)

              ##### compute loss #####
              loss = criterion(output, labels)

              ##### compute classification accuracy #####
              acc = torch.sum( pred == labels) / bs

              batch_losses.append( loss.item() )
              batch_acc.append( acc.item() )

        mean_test_losses.append( np.mean(batch_losses) )
        mean_test_acc.append( np.mean(batch_acc) )

        print( 'Iter %d/%d - Train Loss: %.3f - Test Loss: %.3f - Train acc: %.3f - Test acc: %.3f'
                % (epoch + 1, max_epoch, mean_train_losses[epoch], mean_test_losses[epoch],
                   mean_train_acc[epoch], mean_test_acc[epoch]) )


        # save models
        mod_path = model_dir + 'epoch{:03d}.pt'.format(epoch+1)

        if enc != 0:
            torch.save({
                        'enc_state_dict': enc.state_dict(),
                        'model_state_dict': model.state_dict(),
                        'dec_state_dict': dec.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, mod_path)
        else:
            torch.save({
                        'model_state_dict': model.state_dict(),
                        'dec_state_dict': dec.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, mod_path)

        # save stats.
        stats_path = model_dir + 'stats.csv'
        stats = pd.DataFrame()
        stats["mean train losses"] = mean_train_losses
        stats["mean test losses"] = mean_test_losses
        stats["mean train acc"] = mean_train_acc
        stats["mean test acc"] = mean_test_acc
        stats.to_csv(stats_path, index=False)

        if device.type == 'cuda':
          torch.cuda.empty_cache()

    if enc != 0:
        return enc, model, dec, stats
    else:
        return model, dec, stats
    


def main():
        
        # Hardware agnostic settings.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device in use is: ", device)


        # Import fashion MNIST and create dataloaders.

        # zero-center and normalize the distribution of the image tile content
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        # Create datasets for training & validation, download if necessary
        train_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
        test_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

        # Create data loaders for our datasets; shuffle for train, not for test.
        # Batches have shape (bs, 1, 28, 28)
        bs = 250
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False)

        # Class labels
        classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

        n_classes = len(classes)

        # Report split sizes
        print( 'Training set has {} instances'.format( len(train_set) ) )
        print( 'Testing set has {} instances'.format( len(test_set) ) )


        # Training loop.

        # directory settings.
        exp_no = 1 # experiment number.
        model_name = 'LurieNet_k'
        model_dir = root + 'Models/FMNIST/' + model_name + '/Exp_{:03d}/'.format(exp_no)


        # Model settings.
        d = 28
        q = 1 # number of graph coupled networks.
        n = 576  # state dimension.
        tmax = 100 # Number of time steps to loop through.
        step = 0.01 # integration step.
        g = 1 # upper bound on slope of nonlinearity.
        k = 3
        A_type = 'sym' # full, sym, diag
        B = None
        C = None
        bx = None
        by = None

        # Training settings.
        min_epoch = 0
        max_epoch = 50
        lr = 1e-3
        wd = 1e-5
        decay = 5e-1
        decay_epochs = [35]
        stop_freeze = 0

        ###################################################################################################

        # don't overwrite models
        if os.path.isdir(model_dir):
            raise Exception("File already exists! Do not overwrite")
        else:
            os.makedirs(model_dir)

        # instantiate model
        if model_name == 'LurieNet_k':
            print('LurieNet_k chosen as model!')
            enc = CNN.CNN().to(device)
            model = LN_k.LurieNet_k(q*n, tmax, step, g, k, A_type, B, C, bx, by).to(device)
            dec = dc.Decoder(q*n, n_classes).to(device)
        elif model_name == 'LurieNet':
            print('LurieNet chosen as model!')
            enc = CNN.CNN().to(device)
            model = LN.LurieNet(q*n, tmax, step, A_type, B, C, bx, by).to(device)
            dec = dc.Decoder(q*n, n_classes).to(device)
        elif model_name == 'GLN_k':
            print('GLN chosen as model!')
            # enc = Encoder(d*d, q*n).to(device)
            model = GLN.GLN(q, n, tmax, step, g, k, A_type, B, C, bx, by).to(device)
            dec = dc.Decoder(q*n, n_classes).to(device)


        # save model and training run settings
        model_params = {'q':q, 'n':n, 'tmax':tmax, 'step':step, 'g':g, 'k':k, 'A_type':A_type,
                        'B':B, 'C':C, 'bx':bx, 'by':by}

        run_settings = {'bs':bs, 'max_epoch':max_epoch, 'lr':lr, 'wd':wd, 'decay':decay,
                'decay_epochs':decay_epochs, 'stop_freeze':stop_freeze}


        with open(model_dir + 'run_settings.pkl', 'wb') as f:
            pickle.dump(run_settings, f)
            f.close()
        with open(model_dir + 'model_params.pkl', 'wb') as f:
            pickle.dump(model_params, f)
            f.close()


        enc_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        mod_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        dec_params = sum(p.numel() for p in dec.parameters() if p.requires_grad)

        print('total parameter count =',  enc_params + mod_params + dec_params)
        print('LN parameter count =',  mod_params + dec_params)

        # instantiate loss and optimizer
        criterion = nn.CrossEntropyLoss()


        optimizer = torch.optim.Adam(list(enc.parameters()) + list(model.parameters()) + list(dec.parameters()),
                                     lr=lr, weight_decay=wd)

        start_time = time.time()

        # Train model.
        enc, model, dec, stats = train(train_loader, test_loader, max_epoch, model, dec, optimizer,
                                             criterion, decay, decay_epochs, model_dir, device, enc=enc, stop_freeze=stop_freeze)


        exe_time = time.time() - start_time
        print( "--- %s seconds ---" % (exe_time) )

        # save training run settings after run to include exe_time
        run_settings = {'bs':bs, 'max_epoch':max_epoch, 'lr':lr, 'wd':wd, 'decay':decay,
                'decay_epochs':decay_epochs, 'exe_time':exe_time, 'stop_freeze':stop_freeze}

        with open(model_dir+'run_settings.pkl', 'wb') as f:
            pickle.dump(run_settings, f)
            f.close()



if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import sys
import os
import time



root = ''
sys.path.append(root + 'Code/')
sys.path.append(root + 'Code/Models')
sys.path.append(root + 'Code/Train')
sys.path.append(root + 'Code/Utils')

import Plots_Tools as PT
import Train_Tools as Train
import kLurieNet_Hurwitz as LN_k



def main():

    ### Hardware agnostic settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in use is: ", device)

    # Root directory.
    root = ''

    # Dataset.
    task = 'Opinion' # Opinion, Hopfield, Attractor.
    D = '1a' # Dataset number.
    data_path = root + 'Data/' + task + '/Dataset{0}/'.format(D)

    # Data variables.
    batch = 10
    bs = 100
    test_split = 0.1
    rseed = np.random.randint(0,100)
    n = 3 # state dimension
    q = 1 # number of graph coupled networks.
    k = 1

    # Model.
    exp_no = 1 # experiment number
    model_name = 'LurieNet_k/'
    model_dir = root + 'Models/' + model_name + 'Exp_{:03d}/'.format(exp_no)

    # Training.
    max_epoch = 100
    lr = 5e-3
    wd = 1e-5
    decay = 0
    decay_epochs = []

    ######################## all user settings specified above ######################## 

    if os.path.isdir(model_dir):
        raise Exception("File already exists! Do not overwrite")
    else:
        os.makedirs(model_dir)

    torch.manual_seed(rseed)

    # Load data (batch, bs, tmax, n).
    X_train, X_test, T, exp_setup = Train.get_data(data_path, batch, bs, test_split, rseed, device)

    _, _, step = exp_setup['time']

    _, _, tmax, _ = X_test.shape # (test_batch, bs, tmax, n)

    # Instantiate model.
    if model_name == 'LurieNet/':
        print('LurieNet chosen as model!')
        bx = None
        by = None
        model_params = {'q':q, 'n':n, 'tmax':tmax, 'step':step, 'bx':bx, 'by':by}
        model = LN.LurieNet(q*n, tmax, step, bx, by)
    elif model_name == 'LurieNet_k/':
        print('LurieNet_k chosen as model!')
        A_type = 'full'
        B = None
        C = None
        bx = None
        by = None
        g = 1.0
        model_params = {'q':q, 'n':n, 'tmax':tmax, 'step':step, 'g':g, 'k':k, 'A_type':A_type, 'B':B, 'C':C, 'bx':bx, 'by':by}
        model = LN_k.LurieNet_k(q*n, tmax, step, g, k, A_type, B, C, bx, by)
    elif model_name == 'GLN/':
        print('GLN chosen as model!')
        A_type = 'full'
        B = None
        C = None
        bx = None
        by = None
        g = 1.0
        model_params = {'q':q, 'n':n, 'tmax':tmax, 'step':step, 'g':g, 'k':k, 'A_type':A_type, 'B':B, 'C':C, 'bx':bx, 'by':by}
        model = GLN.GLN(q, n, tmax, step, g, k, A_type, B, C, bx, by)
    elif model_name == 'LipschitzRNN/':
        print('LipschitzRNN chosen as model!')
        by = None
        model_params = {'q':q, 'n':n, 'tmax':tmax, 'step':step, 'by':by}
        model = LRNN.LipschitzRNN(q*n, tmax, step, by)
    elif model_name == 'ARNN/':
        print('ARNN chosen as model!')
        by = None
        model_params = {'q':q, 'n':n, 'tmax':tmax, 'step':step, 'by':by}
        model = ARNN.AntisymmetricRNN(q*n, tmax, step, by)
    elif model_name == 'SVD_Combo/':
        print('SVD_Combo chosen as model!')
        g = 1.0
        bx = None
        model_params = {'q':q, 'n':n, 'tmax':tmax, 'step':step, 'g':g, 'bx':bx}
        model = SVDC.SVD_Combo(q*n, tmax, step, g, bx)
    elif model_name == 'GC_SVDC/':
        print('GC_SVDC chosen as model!')
        g = 1.0
        bx = None
        model_params = {'q':q, 'n':n, 'tmax':tmax, 'step':step, 'g':g, 'bx':bx}
        model = GC_SVDC.GC_SVDC(q, n, tmax, step, g, bx)
    elif model_name == 'NeuralODE/':
        print('NeuralODE chosen as model!')
        h = 100
        model_params = {'q':q, 'n':n, 'h':h}
        model = NODE.ODEBlock( NODE.ODEFunc(q*n, h) )


    run_settings = {'D':D, 'batch':batch, 'bs':bs, 'test_split':test_split, 'rseed':rseed, \
                    'max_epoch':max_epoch, 'lr':lr, 'wd':wd, 'decay':decay, 'decay_epochs':decay_epochs}
        
    # Save run and model settings.
    with open(model_dir+'run_settings.pkl', 'wb') as f:
        pickle.dump(run_settings, f)
        f.close()
    with open(model_dir+'model_params.pkl', 'wb') as f:
        pickle.dump(model_params, f)
        f.close()

    print( 'parameter count =', sum(p.numel() for p in model.parameters() if p.requires_grad) )
    
    # Instantiate optimizer and loss.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    # Train.
    start_time = time.time()
    print( "--- %s seconds ---" % (start_time) )

    model = model.to(device)
    model, stats = Train.train(X_train, X_test, T, model, optimizer, criterion, max_epoch, model_dir,
                decay, decay_epochs, model_name, device)

    exe_time = time.time() - start_time
    print( "--- %s seconds ---" % (exe_time) )


    run_settings = {'D':D, 'batch':batch, 'bs':bs, 'test_split':test_split, 'rseed':rseed, \
                    'max_epoch':max_epoch, 'lr':lr, 'wd':wd, 'decay':decay, 'decay_epochs':decay_epochs,
                    'exe_time':exe_time}

    with open(model_dir+'run_settings.pkl', 'wb') as f:
        pickle.dump(run_settings, f)
        f.close()

    # Plot and save loss.
    file = 'mse'
    PT.plot_mse(model_dir, file, stats, max_epoch)


    return 0



if __name__ == '__main__':
    main()

import pickle
import pandas as pd
import numpy as np


"""
Functions for comparing models.
"""



def loss_multiple_runs(stats_list):
    """
    Compute mean and standard deviation of loss curves of the same model across multiple runs.
    args:
            stats_list: List of dataframes.
    returns:
          stats_av: List of dataframes with mean and standard deviation for training and 
                    testing at each epoch.    
    """

    train_av = pd.DataFrame()
    test_av = pd.DataFrame()
    train = "train loss"
    test = "test loss"
    max_epoch = stats_list[0][train].shape[0]

    for i in range( len(stats_list) ):
        train_i = train + " {:03d}".format(i+1)
        test_i = test + " {:03d}".format(i+1)
        train_av[train_i] = stats_list[i][train]
        test_av[test_i] = stats_list[i][test]

    print('Worst test loss at the last epoch was: ', np.max( test_av.iloc[[max_epoch-1]] ) )
    print( 'Best test loss at the last epoch was: ', np.min( test_av.iloc[[max_epoch-1]] ) )
    
    train_av["average"] = train_av.mean(axis=1)
    test_av["average"] = test_av.mean(axis=1)

    train_av["standard"] = train_av.std(axis=1)
    test_av["standard"] = test_av.std(axis=1)

    return [train_av, test_av]



def model_stats(D:int, model_dir:str, ex:str):
    """
    Returns the model stats for a specified model and total number of experiments.
    args:
               D: Number of experiments.
       model_dir: Location of stats.
              ex: Example.
            
    returns:
            List of dataframes with mean and standard deviation for training and 
            testing at each epoch.
    """

    stats = []
    
    for i in range(1,D+1):
        stats_loc = model_dir + ex + '/Exp_{:03d}/'.format(i)
        stats.append( pd.read_csv(stats_loc + 'stats.csv') ) 

    return loss_multiple_runs(stats)


    
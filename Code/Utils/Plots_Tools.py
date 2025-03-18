import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



"""
A set of plotting functions used by data generation and training scripts. Also used for creating 
figures in paper.
"""



def plot3d(loc:str, file:str, X1:np.array, X2:np.array, X3:np.array, eq1:np.array, eq2:np.array, q1:int, q3:int, title1:str, title2:str, title3:str, min_axis1:list, max_axis1:list, min_axis2:list, max_axis2:list, min_axis3:list, max_axis3:list):
    """
    Plots a random batch of trajectories in 3d for k=1,2,3 contracting example systems.
    args:
              loc: location to save figure.
             file: name to save figure as.
                X1: trajectory data.
                X2: trajectory data.
                X3: trajectory data.
               eq1: array containing 1 equilibrium points (1,3).
               eq2: array containing 3 equilibrium points (3, 3).
                q1: number of trajectories to plot for X1 and X2.
                q3: number of trajectories to plot for X3. 
            title1: title to give figure 1.
            title2: title to give figure 2.
            title3: title to give figure 3.
         min_axis1: lower bounds of axes.
         max_axis1: upper bounds of axes.
         min_axis2: lower bounds of axes.
         max_axis2: upper bounds of axes.
         min_axis3: lower bounds of axes.
         max_axis3: upper bounds of axes.
    """

    tmax, N, n = X1.shape

    # randomly pick a batch of trajectories to plot.
    a = np.random.randint(N, size=q1)
    a3 = np.random.randint(N, size=q3)

    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    for i in range(q1):
        ax1.plot( X1[:,a[i],0], X1[:,a[i],1], X1[:,a[i],2] )
        ax1.plot(X1[0,a[i],0], X1[0,a[i],1], X1[0,a[i],2], 'xr')
        ax2.plot( X2[:,a[i],0], X2[:,a[i],1], X2[:,a[i],2] )
        ax2.plot(X2[0,a[i],0], X2[0,a[i],1], X2[0,a[i],2], 'xr')

    for i in range(q3):
        ax3.plot( X3[:,a3[i],0], X3[:,a3[i],1], X3[:,a3[i],2] )
        ax3.plot(X3[0,a3[i],0], X3[0,a3[i],1], X3[0,a3[i],2], 'xr')
        
    # plot equilibrium points
    ax1.plot(eq1[0], eq1[1], eq1[2], '*r')
    ax2.plot(eq2[0,0], eq2[0,1], eq2[0,2], '*r')
    ax2.plot(eq2[1,0], eq2[1,1], eq2[1,2], '*r')
    ax2.plot(eq2[2,0], eq2[2,1], eq2[2,2], '*r')

    # axes limits
    ax1.set_xlim(min_axis1[0], max_axis1[0])
    ax1.set_ylim(min_axis1[1], max_axis1[1])
    ax1.set_zlim(min_axis1[2], max_axis1[2])
    ax2.set_xlim(min_axis2[0], max_axis2[0])
    ax2.set_ylim(min_axis2[1], max_axis2[1])
    ax2.set_zlim(min_axis2[2], max_axis2[2])
    
    # axes titles
    ax1.set_title(title1)
    ax2.set_title(title2)
    ax3.set_title(title3)
    
    # axes labels
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("x3")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_zlabel("x3")
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_zlabel("x3")

    fig.savefig(loc + file + '.png')
    plt.show()

    return 0 
    


def plot_mse(loc:str, file:str, stats:pd.DataFrame, max_epoch:int, y_lim=None):
    """
    Plot the mse training and test loss of a single run. 
    args:
        loc: location to save figure.
       file: to save figure as.
      stats: data to plot.
  max_epoch: for creating x axis data.
      y_lim: list containing the upper and lower bounds of y-axis.
    """

    epoch = np.arange(1, max_epoch+1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    
    ax1.plot(epoch, stats['train loss'], label='train')
    ax1.plot(epoch, stats['test loss'], label='test')
    
    if y_lim != None:
        ax1.set_ylim(y_lim)
    ax1.set_xlabel('Epoch')
    ax1.set_title('MSE loss')
    ax1.legend()

    ax2.plot(epoch, np.log(stats['train loss']), label='train')
    ax2.plot(epoch, np.log(stats['test loss']), label='test')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Log plot of MSE loss')
    ax2.legend()

    fig.savefig(loc + file + '.png')
    plt.show()

    return 0



def compare_all_traj(D:str, loc:str, file:str, titles:list, X0:torch.tensor, X1:torch.tensor, X2:torch.tensor, 
                 X3:torch.tensor, X4:torch.tensor, X5:torch.tensor, X6:torch.tensor, q:int, eqs=[]):
    """
    Plot trajectories from dataset X against predictions X2 to compare against each other.
    This will only work for 3d data.
    args:
        D: Ground truth dataset number.
      loc: location to save figure.
     file: Name to save figure as.
   titles: List of titles for each dataset.
       X0: Dataset 0.
       X1: Dataset 1.
       X2: Dataset 2.
       X3: Dataset 3.
       X4: Dataset 4.
       X5: Dataset 5.
       X6: Dataset 6.
        q: Number of trajectories to compare.
      eqs: (optional) Equilibrium points of ground truth system if they exist.
    """

    bs, tmax, n = X0.shape

    # randomly pick q trajectories to compare.
    a = torch.randint(0, bs, (q,))

    fig = plt.figure(figsize=(14,7))
    fig.tight_layout()
    
    ax0 = fig.add_subplot(2, 4, 1, projection='3d')
    ax1 = fig.add_subplot(2, 4, 2, projection='3d')
    ax2 = fig.add_subplot(2, 4, 3, projection='3d')
    ax3 = fig.add_subplot(2, 4, 4, projection='3d')
    ax4 = fig.add_subplot(2, 4, 5, projection='3d')
    ax5 = fig.add_subplot(2, 4, 6, projection='3d')
    ax6 = fig.add_subplot(2, 4, 7, projection='3d')
    
    for i in range(q):
        j=a[i]
        ax0.plot(X0[j,:,0], X0[j,:,1], X0[j,:,2])
        ax0.plot(X0[j,0,0], X0[j,0,1], X0[j,0,2], 'xr')
        ax1.plot(X1[j,:,0], X1[j,:,1], X1[j,:,2])
        ax1.plot(X1[j,0,0], X1[j,0,1], X1[j,0,2], 'xr')       
        ax2.plot(X2[j,:,0], X2[j,:,1], X2[j,:,2])
        ax2.plot(X2[j,0,0], X2[j,0,1], X2[j,0,2], 'xr')
        ax3.plot(X3[j,:,0], X3[j,:,1], X3[j,:,2])
        ax3.plot(X3[j,0,0], X3[j,0,1], X3[j,0,2], 'xr')
        ax4.plot(X4[j,:,0], X4[j,:,1], X4[j,:,2])
        ax4.plot(X4[j,0,0], X4[j,0,1], X4[j,0,2], 'xr')
        ax5.plot(X5[j,:,0], X5[j,:,1], X5[j,:,2])
        ax5.plot(X5[j,0,0], X5[j,0,1], X5[j,0,2], 'xr')
        ax6.plot(X6[j,:,0], X6[j,:,1], X6[j,:,2])
        ax6.plot(X6[j,0,0], X6[j,0,1], X6[j,0,2], 'xr')

    # plot equilibrium points
    if ( D == '1a' or D == '1b' ) and (len(eqs)!=0) :
        ax0.plot(eqs[0], eqs[1], eqs[2], '*r')
        ax1.plot(eqs[0], eqs[1], eqs[2], '*r')
        ax2.plot(eqs[0], eqs[1], eqs[2], '*r')
        ax3.plot(eqs[0], eqs[1], eqs[2], '*r')
        ax4.plot(eqs[0], eqs[1], eqs[2], '*r')
        ax5.plot(eqs[0], eqs[1], eqs[2], '*r')
        ax6.plot(eqs[0], eqs[1], eqs[2], '*r')
    elif ( D == '2a' or D == '2b' ) and (len(eqs)!=0) :
        ax0.plot(eqs[0,0], eqs[0,1], eqs[0,2], '*r')
        ax0.plot(eqs[1,0], eqs[1,1], eqs[1,2], '*r')
        ax0.plot(eqs[2,0], eqs[2,1], eqs[2,2], '*r')
        ax1.plot(eqs[0,0], eqs[0,1], eqs[0,2], '*r')
        ax1.plot(eqs[1,0], eqs[1,1], eqs[1,2], '*r')
        ax1.plot(eqs[2,0], eqs[2,1], eqs[2,2], '*r')
        ax2.plot(eqs[0,0], eqs[0,1], eqs[0,2], '*r')
        ax2.plot(eqs[1,0], eqs[1,1], eqs[1,2], '*r')
        ax2.plot(eqs[2,0], eqs[2,1], eqs[2,2], '*r')
        ax3.plot(eqs[0,0], eqs[0,1], eqs[0,2], '*r')
        ax3.plot(eqs[1,0], eqs[1,1], eqs[1,2], '*r')
        ax3.plot(eqs[2,0], eqs[2,1], eqs[2,2], '*r')
        ax4.plot(eqs[0,0], eqs[0,1], eqs[0,2], '*r')
        ax4.plot(eqs[1,0], eqs[1,1], eqs[1,2], '*r')
        ax4.plot(eqs[2,0], eqs[2,1], eqs[2,2], '*r')
        ax5.plot(eqs[0,0], eqs[0,1], eqs[0,2], '*r')
        ax5.plot(eqs[1,0], eqs[1,1], eqs[1,2], '*r')
        ax5.plot(eqs[2,0], eqs[2,1], eqs[2,2], '*r')
        ax6.plot(eqs[0,0], eqs[0,1], eqs[0,2], '*r')
        ax6.plot(eqs[1,0], eqs[1,1], eqs[1,2], '*r')
        ax6.plot(eqs[2,0], eqs[2,1], eqs[2,2], '*r')
    
    # ax0.set_xlabel("x1")
    # ax0.set_ylabel("x2")
    # ax0.set_zlabel("x3")
    ax0.set_title(titles[0])

    # ax1.set_xlabel("x1")
    # ax1.set_ylabel("x2")
    # ax1.set_zlabel("x3")
    ax1.set_title(titles[1])

    # ax2.set_xlabel("x1")
    # ax2.set_ylabel("x2")
    # ax2.set_zlabel("x3")
    ax2.set_title(titles[2])

    # ax3.set_xlabel("x1")
    # ax3.set_ylabel("x2")
    # ax3.set_zlabel("x3")
    ax3.set_title(titles[3])

    # ax4.set_xlabel("x1")
    # ax4.set_ylabel("x2")
    # ax4.set_zlabel("x3")
    ax4.set_title(titles[4])

    # ax5.set_xlabel("x1")
    # ax5.set_ylabel("x2")
    # ax5.set_zlabel("x3")
    ax5.set_title(titles[5])

    # ax6.set_xlabel("x1")
    # ax6.set_ylabel("x2")
    # ax6.set_zlabel("x3")
    ax6.set_title(titles[6])

    fig.savefig(loc + file + '.png')
    plt.show()

    return 0



def compare_traj(D:str, loc:str, file:str, X:torch.tensor, X2:torch.tensor, eqs:torch.tensor, q:int, 
                 x_lim=[], y_lim=[], z_lim=[]):
    """
    Plot trajectories from dataset X against predictions X2 to compare against each other.
    This will only work for 3d data.
    args:
        D: Ground truth dataset number.
      loc: location to save figure.
     file: Name to save figure as. 
        X: Dataset 1.
       X2: Dataset 2.
      eqs: Equilibrium points of ground truth system.
        q: Number of trajectories to compare.
    """

    batch, bs, tmax, n = X.shape

    # randomly pick batches and trajectories
    a = torch.randint(0, batch, (q,))
    b = torch.randint(0, bs, (q,))

    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.tight_layout()
    
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    
    for i in range(q):
        k=a[i]
        j=b[i]
            
        ax0.plot(X[k,j,:,0], X[k,j,:,1], X[k,j,:,2])
        ax0.plot(X[k,j,0,0], X[k,j,0,1], X[k,j,0,2], 'xr')
        ax1.plot(X2[k,j,:,0], X2[k,j,:,1], X2[k,j,:,2])
        ax1.plot(X2[k,j,0,0], X2[k,j,0,1], X2[k,j,0,2], 'xr')


    # plot equilibrium points
    if D == '1a' or D == '1b':
        ax0.plot(eqs[0], eqs[1], eqs[2], '*r')
        ax1.plot(eqs[0], eqs[1], eqs[2], '*r')
    elif D == '2a' or D == '2b':
        ax0.plot(eqs[0,0], eqs[0,1], eqs[0,2], '*r')
        ax0.plot(eqs[1,0], eqs[1,1], eqs[1,2], '*r')
        ax0.plot(eqs[2,0], eqs[2,1], eqs[2,2], '*r')
        ax1.plot(eqs[0,0], eqs[0,1], eqs[0,2], '*r')
        ax1.plot(eqs[1,0], eqs[1,1], eqs[1,2], '*r')
        ax1.plot(eqs[2,0], eqs[2,1], eqs[2,2], '*r')

    if len(x_lim) != 0:
        ax0.set_xlim(x_lim)
        ax1.set_xlim(x_lim)

    if len(y_lim) != 0:
        ax0.set_ylim(y_lim)
        ax1.set_ylim(y_lim)

    if len(z_lim) != 0:
        ax0.set_zlim(z_lim)
        ax1.set_zlim(z_lim)

    ax0.set_xlabel("x1")
    ax0.set_ylabel("x2")
    ax0.set_zlabel("x3")
    ax0.set_title("Ground Truth")

    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("x3")
    ax1.set_title("Predictions")
    
    fig.savefig(loc + file + '.png')
    plt.show()

    return 0



def plot_mse_all_stats(loc:str, file:str, all_stats_ex1, all_stats_ex2):
    """
    Plot the mean and 2 standard deviation interval of training and test losses 
    for different models and tasks. Note that lower interval limit is cut off at zero. 
    args:
              loc: location to save figure.
             file: name to save figure as.
        all_stats_ex1: a list containing training and test loss stats for each model on task 1.
        all_stats_ex2: a list containing training and test loss stats for each model on task 2.
        all_stats_ex3: a list containing training and test loss stats for each model on task 3.
    """

    max_epoch = all_stats_ex1[0][0][0].shape[0]
    epochs = np.arange(1, max_epoch+1)

    q = len(all_stats_ex1)

    # fig, (axs) = plt.subplots(3,q, figsize=(8, 9))
    fig, (axs) = plt.subplots(2,q, figsize=(5, 9))
    fig.tight_layout()
    
    # indices = (model number, 0=data,1=model name, 0=train, 1=test, dataframe column)

    # Task 1
    for i in range(q):

        # train losses
        mean = all_stats_ex1[i][0][0]['average']
        std = all_stats_ex1[i][0][0]['standard']
        low = mean - 2*std
        low[low < 0] = 0
        upp = mean + 2*std
        name = all_stats_ex1[i][1]
        axs[0,i].plot(epochs, mean, label='train')
        axs[0,i].fill_between(epochs, low, upp, alpha=0.5)

        # test losses
        mean = all_stats_ex1[i][0][1]['average']
        std = all_stats_ex1[i][0][1]['standard']
        low = mean - 2*std
        low[low < 0] = 0 # if less than 0, set to zero
        upp = mean + 2*std
        name = all_stats_ex1[i][1]
        axs[0,i].plot(epochs, mean, label='test')
        axs[0,i].fill_between(epochs, low, upp, alpha=0.5)

    # Task 2
    for i in range(q):

        # train losses
        mean = all_stats_ex2[i][0][0]['average']
        std = all_stats_ex2[i][0][0]['standard']
        low = mean - 2*std
        low[low < 0] = 0
        upp = mean + 2*std
        name = all_stats_ex2[i][1]
        axs[1,i].plot(epochs, mean, label='train')
        axs[1,i].fill_between(epochs, low, upp, alpha=0.5)

        # test losses
        mean = all_stats_ex2[i][0][1]['average']
        std = all_stats_ex2[i][0][1]['standard']
        low = mean - 2*std
        low[low < 0] = 0
        upp = mean + 2*std
        name = all_stats_ex2[i][1]
        axs[1,i].plot(epochs, mean, label='test')
        axs[1,i].fill_between(epochs, low, upp, alpha=0.5)


    # # Task 3
    # for i in range(q):

    #     # train losses
    #     mean = all_stats_ex3[i][0][0]['average']
    #     std = all_stats_ex3[i][0][0]['standard']
    #     low = mean - 2*std
    #     low[low < 0] = 0
    #     upp = mean + 2*std
    #     name = all_stats_ex3[i][1]
    #     axs[2,i].plot(epochs, mean, label='train')
    #     axs[2,i].fill_between(epochs, low, upp, alpha=0.5)

    #     # test losses
    #     mean = all_stats_ex3[i][0][1]['average']
    #     std = all_stats_ex3[i][0][1]['standard']
    #     low = mean - 2*std
    #     low[low < 0] = 0
    #     upp = mean + 2*std
    #     name = all_stats_ex3[i][1]
    #     axs[2,i].plot(epochs, mean, label='test')
    #     axs[2,i].fill_between(epochs, low, upp, alpha=0.5)


    axs[0,2].yaxis.set_major_formatter('{x:.02f}') 
    axs[0,4].yaxis.set_major_formatter('{x:.02f}') 

    # axs[0,0].set_ylim(bottom=0, top=0.02)
    # axs[0,1].set_ylim(bottom=0, top=0.1)
    # axs[0,2].set_ylim(bottom=0, top=0.02)
    # axs[0,3].set_ylim(bottom=0, top=1.0)
    # axs[0,4].set_ylim(bottom=0, top=0.02)
    # axs[1,1].set_ylim(bottom=0, top=1.0)
    # axs[1,3].set_ylim(bottom=0, top=1.2)
    # axs[1,5].set_ylim(bottom=0.3, top=0.7)
    # axs[2,1].set_ylim(bottom=0, top=5.5)
    # axs[2,3].set_ylim(bottom=0, top=5)

    axs[0,1].set_ylim(bottom=2, top=25.0)
    axs[0,2].set_ylim(bottom=0, top=1.5)
    axs[0,3].set_ylim(bottom=0.1, top=0.6)
    axs[0,4].set_ylim(bottom=0, top=2.0)
    axs[0,5].set_ylim(bottom=0.4, top=0.75)
    axs[1,0].set_ylim(bottom=0, top=2.5)
    axs[1,1].set_ylim(bottom=0, top=100.0)
    axs[1,2].set_ylim(bottom=2.5, top=6.0)
    axs[1,3].set_ylim(bottom=0, top=4.0)
    axs[1,4].set_ylim(bottom=2.0, top=6.5)

    # axs[0,0].set_title('k Lurie Net.')
    # axs[0,1].set_title('Lurie Net.')
    # axs[0,2].set_title('Neural ODE')
    # axs[0,3].set_title('Lipschitz')
    # axs[0,4].set_title('SVD Combo')
    # axs[0,5].set_title('Antisym.')

    axs[0,0].set_title('GLN')
    axs[0,1].set_title('Lurie')
    axs[0,2].set_title('ODE')
    axs[0,3].set_title('Lipschitz')
    axs[0,4].set_title('GC SVD')
    axs[0,5].set_title('Antisym.')
    axs[0,6].set_title('k Lurie')

    # axs[0,0].set_ylabel('Average MSE Opinion')
    # axs[1,0].set_ylabel('Average MSE Hopfield')
    # axs[2,0].set_ylabel('Average MSE Attractor')

    axs[0,0].set_ylabel('Average MSE GC Hopfield')
    axs[1,0].set_ylabel('Average MSE GC Attractor')

    plt.subplots_adjust(left=0.1)
    
    fig.savefig(loc + file + '.png')
    plt.show()

    return 0



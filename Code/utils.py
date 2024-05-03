def random_lambda(n:int, k:int, g:float, gb:float, gc:float):
    """Function which generates a random vector for SigmaA1 when k>1,
        subject to the k-contraction constraint - refer to Algorithm 1.
        args:
            n: dimension of the vector.
            k: integer between 2 and n which controls the dimension of the k-compound system.
            g: positive real value describing the upper bound on the slope of the nonlinearity.
           gb: positive real value describing the upper bound on the sum of the k-largest singular values of the B matrix.
           gc: positive real value describing the upper bound on the sum of the k-largest singular values of the C matrix.
        returns:
                square n x n matrix with the eigenvalues of the symmetric component of A (which satisfy Theorem 2) along the main diagonal.
    """
    if k<2:
        raise ValueError(f"This function should only be used for k>1!")

    sk = -2*g*gb*gc/k + 1 # +1 is just to satisfy while loop conditions
    sk_1 = -2*g*gb*gc/(k-1) - 1 # -1 is just to satisfy while loop conditions
    r1 = -2*g*gb*gc/((k-1)**2)
    r2 = 2*g*gb*gc/(k**2)
    r_lambda = torch.zeros(n)

    while sk >= -2*g*gb*gc/k or sk_1 < -2*g*gb*gc/(k-1):
        r_lambda[0:k] = torch.sort((r2 - r1)*torch.rand(k) + r1*torch.ones(k),descending=True).values # ~ U(r1,r2)
        sk_1 = torch.sum(r_lambda[0:k-1])
        sk = sk_1 + r_lambda[k-1]

    r_lambda[k:n] = r_lambda[k-1]*torch.ones(n-k) - torch.rand(n-k)

    return torch.diag(r_lambda)

def get_init(output_size:int, input_size:int, init:str, std=0.1):
    """Returns a random initialisation following using the scheme init.
        Initialisation is only for weight matrices, bias' are set to zero.
        args:
            output_size = dimension of matrix output.
            input_size = dimension of matrix input.
            std: standard deviation of normal distribution.
           init: the initialisation scheme to use.
        returns: tensor of shape [output_size, input_size] sampled from specified normal distribution.
        """
    if init=='He':
        return torch.normal(0,2*std/input_size,(output_size,input_size))
    elif init=='Xavier':
        return torch.normal(0,std/input_size,(output_size,input_size)) # used in LipschitzRNN
    elif init=='RNNsofRNNs':
        return torch.normal(0,std/np.sqrt(input_size), (output_size,input_size))
    elif init=='He+RNNsofRNNs':
        return torch.normal(0,2*std/np.sqrt(input_size), (output_size,input_size)) # made up
    else:
        raise ValueError(f"Unspecified initialisation.")
    

def add_channels(X):
    """ Helper function when permutting input data for psmnist task. Dependent on the shape of the permutted dataset,
        the dataset is reshaped.
        args:
            X: permutted dataset
     returns: reshaped permutted dataset
    """
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], 1, X.shape[1], 1)
    elif len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    else:
        return "Dataset has unexpected dimensions!"
    
def lr_scheduler(epoch:int, optimizer, decay_eff:float, decayEpoch):
    """Decay learning rate by a factor of decay_eff every epoch is equal to an element in decayEpoch.
        args:
            epoch: current epoch to compare against.
            optimizer: the optimizer for which we wish to update the learning rate.
            decay_eff: the factor to multiply the current learning rate by.
            decayEpoch: list of epoch's which the learning rate should be decreased at.
        returns: optimizer with updated learning rate.
        """
    if epoch in decayEpoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_eff
        print('New learning rate is: ', param_group['lr'])

    return optimizer



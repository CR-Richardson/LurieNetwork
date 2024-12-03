import torch
import torch.nn as nn



class LurieNet(nn.Module):
    """
    Unconstrained Lurie Network.
    args:
            n: Dimension of state.
         tmax: Number of time steps to loop through.
         step: Integration step size.
           bx: None if bx is to be learnt, or a tensor with dtype double if bx is known.
           by: None if by is to be learnt, or a tensor with dtype double if by is known.
    returns:
            X: Predictions for a given batch with size (bs, tmax, n).
    """
    
    def __init__(self, n:int, tmax:int, step:float, bx, by):
        super().__init__()
        self.n = n
        self.tmax = tmax
        self.step = step

        self.A_ = nn.Linear(n, n, bias=False, dtype=torch.double)
        self.B_ = nn.Linear(n, n, bias=False, dtype=torch.double)
        self.C_ = nn.Linear(n, n, bias=False, dtype=torch.double)
        nn.init.normal_(self.A_.weight, mean=-1.0, std=0.1)
        nn.init.normal_(self.B_.weight, mean=0., std=0.1)
        nn.init.normal_(self.C_.weight, mean=0., std=0.1)

        if bx == None:
            self.bx_ind = True
            self.bx_ = nn.Linear(1,n, bias=False, dtype=torch.double)
            nn.init.zeros_(self.bx_.weight)
        else:
            self.bx_ind = False
            self.bx_ = bx
            
        if by == None:
            self.by_ind = True
            self.by_ = nn.Linear(1,n, bias=False, dtype=torch.double)
            nn.init.zeros_(self.by_.weight)
        else:
            self.by_ind = False
            self.by_ = by
          

    def forward(self, X0):

        bs, n = X0.shape

        if self.by_ind == True:
            self.by = torch.tile( self.by_.weight, (1,bs) )
        else:
            self.by = torch.tile( self.by_, (1,bs) )

        if self.bx_ind == True:
            self.bx = torch.tile( self.bx_.weight, (1,bs) )
        else:
            self.bx = torch.tile( self.bx_, (1,bs) )
        
        self.A = self.A_.weight
        self.B = self.B_.weight
        self.C = self.C_.weight
        
        X = []
        Xi = X0.t() # (n, bs)
        X.append(Xi)
    
        for i in range(self.tmax-1):
            Yi = torch.matmul( self.C, Xi ) + self.by
            Xi = Xi + self.step * ( torch.matmul( self.A, Xi )  + torch.matmul( self.B, torch.tanh(Yi)  ) + self.bx )
            X.append(Xi)

        X = torch.stack(X, dim=0) # (tmax, n, bs)
        X = torch.transpose(X,0,2) # (bs, n, tmax)
        
        return torch.transpose(X,1,2) # (bs, tmax, n)




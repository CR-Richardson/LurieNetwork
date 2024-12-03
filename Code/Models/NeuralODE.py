import torch
import torch.nn as nn
# from torchdiffeq import odeint # used on small scale examples
from torchdiffeq import odeint_adjoint as odeint # needed to speed up integration for graph examples



"""
Implementation of Neural ODE from 'Neural ordinary differential equations', Chen et. al. (2018) with same structure as used
in 'Heavy ball neural ordinary differential equations' Xia et. al. (2021).
"""



class ODEFunc(nn.Module):
    """
    Neural ODE model.
    args:
        n: dimension of state.
    """
    
    def __init__(self, n, h):
        super(ODEFunc, self).__init__()
        
        self.fc1 = nn.Linear(n, h, dtype=torch.double)
        self.fc2 = nn.Linear(h, h, dtype=torch.double)
        self.fc3 = nn.Linear(h, h, dtype=torch.double)
        self.fc4 = nn.Linear(h, n, dtype=torch.double)

        self.NODE = nn.Sequential(
                self.fc1,
                nn.ReLU(),
                self.fc2,
                nn.ReLU(),
                self.fc3,
                nn.ReLU(),
                self.fc4
                                )

        for m in self.NODE.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(self, t, X0):
        return self.NODE(X0)



class ODEBlock(nn.Module):
    """
    ODE integrator.
    args:
        odefunc: Neural ODE model.
    """
    
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.func = odefunc

    def forward(self, t, X0):
        return torch.transpose( odeint(self.func, X0, t, method='euler'), 1, 0 ) # (bs, tmax, n)



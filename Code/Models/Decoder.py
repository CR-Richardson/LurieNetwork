import torch
import torch.nn as nn



class Decoder(nn.Module):
    '''
    Decoder. Maps final latent state (bs, n) to logits of class probabilities. (bs, n_classes).
    args:
          input_dim: dimension of latent state (n).
         output_dim: number of classes (n_classes).
    '''

    def __init__(self, input_dim:int, output_dim:int):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer = nn.Linear(input_dim, output_dim, dtype=torch.double)
        nn.init.normal_(self.layer.weight, mean=0, std=1e-2)
        nn.init.constant_(self.layer.bias, val=0)

    def forward(self, x):
        return self.layer(x)
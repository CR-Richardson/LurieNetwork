class LurieNet(nn.Module):
    """Class for constructing the Lurie network.
        args:
            input_size: dimension of the Lurie network input.
           output_size: dimension of the Lurie network output.
                     n: dimension of the Lurie network state (x).
                     m: dimension of the argument to the nonlinearity (y).
                     k: integer in [1,n] which controls the dimension of the k-compound system. This class only works for k=1.
                     g: positive real value describing the upper bound on the slope of the nonlinearity.
                    gb: positive real value describing the upper bound on the sum of the k-largest singular values of the B matrix.
                    gc: positive real value describing the upper bound on the sum of the k-largest singular values of the C matrix.
                   ga1: positive real value for lower bounding the eigenvalues of the symmetric component of A.
                   ga2: positive real value for upper bounding the magnitude of the skew-symmetric component of A.
                 delta: positive real value for discretizing continuous time dynamics.
                v1_ind: boolean variable for indicating if input v1 is present in model.
                v2_ind: boolean variable for indicating if input v2 is present in model.
                  init: string containing the specified weight initialisation.
optional args:
                     A: nxn tensor for passing in a A matrix when it is not being learnt.
                     B: nxm tensor for passing in a B matrix when it is not being learnt.
                     C: mxn tensor for passing in a C matrix when it is not being learnt.
    functions:
            forward: one step computes one step ahead of the Lurie network.
    """

    def __init__(self, input_size, output_size, n, m, k, g, gb, gc, ga1, ga2, delta, v1_ind, v2_ind, init, A=None, B=None, C=None):
        super().__init__()
        self.n = n
        self.m = m
        self.r = min(n,m)
        self.k = k
        self.g = g
        self.gb = gb
        self.gc = gc
        self.ga1 = ga1
        self.ga2 = ga2
        self.delta = delta
        self.v1_ind = v1_ind
        self.v2_ind = v2_ind
        self.A = A
        self.B = B
        self.C = C

        # Instantiate input layer(s) and initialise bias' to zero.
        if v1_ind == True:
            self.V1_in = nn.Linear(input_size, n)
            self.V1_in.bias.data = torch.zeros(self.V1_in.bias.data.shape[0])
        if v2_ind == True:
            self.V2_in = nn.Linear(input_size, m)
            self.V2_in.bias.data = torch.zeros(self.V2_in.bias.data.shape[0])
        if v1_ind == False and v2_ind == False:
            raise ValueError(f"No inputs!.")

        # Instantiate output layer and initialise bias to zero.
        self.out = nn.Linear(n, output_size)
        self.out.bias.data = torch.zeros(output_size)

        # defining number of parameters in skew symmetric matrices and calculating lower triangular indices for nxn and mxm matrices
        skew_n = int(n*(n-1)/2)
        skew_m = int(m*(m-1)/2)
        self.tril_ind_n = torch.tril_indices(n, n, offset=-1) # lower triangular indices
        self.tril_ind_m = torch.tril_indices(m, m, offset=-1) # lower triangular indices

        if C==None:
            # Components of C matrix - UC, VC just need to be skew symmetric, SC just needs to be diagonal with r non-zero elements.
            self.UC = nn.Parameter(get_init(1,skew_m,init))
            self.SC = nn.Parameter(get_init(1,self.r,init))
            self.VC = nn.Parameter(get_init(1,skew_n,init))

        if B==None:
            # Components of B matrix - UB, VB just need to be skew symmetric, SB just needs to be diagonal with r non-zero elements.
            self.UB = nn.Parameter(get_init(1,skew_n,init))
            self.SB = nn.Parameter(get_init(1,self.r,init))
            self.VB = nn.Parameter(get_init(1,skew_m,init))

        if A==None:
            # Components of A matrix  - UA1, UA2 just need to be skew symmetric, SA1 just need to be diagonal with n non-zero elements, and
            # SA2 has a special skew-symmetric form with n-1 non-zero elements.
            self.UA1 = nn.Parameter(get_init(1,skew_n,init))
            self.UA2 = nn.Parameter(get_init(1,skew_n,init))
            self.SA2 = nn.Parameter(get_init(1,n-1,init))
            if k==1:
                self.SA1 = nn.Parameter(get_init(1,n,init))
            elif k>1 and k<=n:
                self.SA1 = random_lambda(n, k, g, gb, gc)
            else:
                raise ValueError(f"Invalid choice of k.")

    def forward(self, inputs):
        """Constructs the LurieNet layer subject to the k-contraction parameterization. Then passes pixels into the network one-by-one, but
           in parrallel for all images in the batch. Hence, vectors x, y, v1, v2 are expressed as matrices below, instead of vectors as described in the paper.
            args:
                self: see above.
                inputs: tensor of shape [batch_size, pixels/image, rgb=input_size]
         returns: tensor of shape [batch size, output size] containing "likelihoods" of each class (prediction = highest likelihood).
        """

        X = torch.zeros(inputs.shape[0], self.n) # shape [batch_size, n]

        if self.C==None:
            # Construct C matrix
            UC_temp = torch.zeros(self.m,self.m)
            UC_temp[self.tril_ind_m[0], self.tril_ind_m[1]] = self.UC # represents self.UC vector as a lower triangular matrix.
            UC = torch.matrix_exp(UC_temp - torch.transpose(UC_temp,0,1)) # Orthogonal parametrization of self.UC
            VC_temp = torch.zeros(self.n,self.n)
            VC_temp[self.tril_ind_n[0], self.tril_ind_n[1]] = self.VC # represents self.VC vector as a lower triangular matrix.
            VC = torch.matrix_exp(VC_temp - torch.transpose(VC_temp,0,1)) # Orthogonal parametrization of self.VC
            Cmask = torch.zeros(self.m, self.n)
            Cmask[0:self.r,0:self.r] = torch.eye(self.r) # Mask to ensure SC has 0 elements for all but r elements on the main diagonal.
            SC_temp = torch.zeros(self.m, self.n)
            SC_temp[[i for i in range(self.r)], [i for i in range(self.r)]] = self.SC # embeds self.SC in an mxn matrix along first r elements on main diagonal.
            SC = (self.gc/self.k)*Cmask*torch.exp(-SC_temp**2) # exp(0)=1, so need mask to set all elements to zero other than first r along main diagonal.
            C = md([UC, SC, torch.transpose(VC,0,1)])
        else:
            C = self.C # Pass in C matrix

        if self.B==None:
            # Construct B matrix
            UB_temp = torch.zeros(self.n,self.n)
            UB_temp[self.tril_ind_n[0], self.tril_ind_n[1]] = self.UB # represents self.UB vector as a lower triangular matrix.
            UB = torch.matrix_exp(UB_temp - torch.transpose(UB_temp,0,1)) # Orthogonal parametrization of self.UB
            VB_temp = torch.zeros(self.m,self.m)
            VB_temp[self.tril_ind_m[0], self.tril_ind_m[1]] = self.VB # represents self.VB vector as a lower triangular matrix.
            VB = torch.matrix_exp(VB_temp - torch.transpose(VB_temp,0,1)) # Orthogonal parametrization of self.VB
            Bmask = torch.zeros(self.n, self.m)
            Bmask[0:self.r,0:self.r] = torch.eye(self.r) # Mask to ensure SB has 0 elements for all but r elements on the main diagonal.
            SB_temp = torch.zeros(self.n, self.m)
            SB_temp[[i for i in range(self.r)], [i for i in range(self.r)]] = self.SB # embeds self.SB in an nxm matrix along first r elements on main diagonal.
            SB = (self.gb/self.k)*Bmask*torch.exp(-SB_temp**2) # exp(0)=1, so need mask to set all elements to zero other than first r along main diagonal.
            B = md([UB, SB, torch.transpose(VB,0,1)])
        else:
            B = self.B # Pass in B matrix

        if self.A==None:
            # Construct A matrix
            UA1_temp = torch.zeros(self.n,self.n)
            UA1_temp[self.tril_ind_n[0], self.tril_ind_n[1]] = self.UA1 # represents self.UA1 vector as a lower triangular matrix.
            UA1 = torch.matrix_exp(UA1_temp - torch.transpose(UA1_temp,0,1)) # Orthogonal parametrization of self.UA1
            UA2_temp = torch.zeros(self.n,self.n)
            UA2_temp[self.tril_ind_n[0], self.tril_ind_n[1]] = self.UA2 # represents self.UA2 vector as a lower triangular matrix.
            UA2 = torch.matrix_exp(UA2_temp - torch.transpose(UA2_temp,0,1)) # Orthogonal parametrization of self.UA2
            A2mask = torch.diag(torch.ones(self.n-1),diagonal=1) # Mask to ensure SA2 has 0 elements for all but n-1 elements above the main diagonal.
            SA2_temp = torch.zeros(self.n,self.n)
            SA2_ind_x = []
            SA2_ind_y = []
            for i in range(self.n): # row
                for j in range(self.n): # col
                    if i-j ==-1:
                        SA2_ind_x.append(i)
                        SA2_ind_y.append(j)
            SA2_temp[SA2_ind_x,SA2_ind_y] = self.SA2 # embeds self.SA2 vector above the main diagonal.
            SA2 = self.ga2*A2mask*torch.exp(-SA2_temp**2) - torch.transpose(self.ga2*A2mask*torch.exp(-SA2_temp**2),0,1) # exp(0)=1, so need mask to set all elements to zero other than those above the main diagonal.
            if self.k == 1:
                A1mask = torch.eye(self.n) # Mask to ensure SA1 has 0 elements for all but n elements along the main diagonal.
                A1_temp = torch.zeros(self.n,self.n)
                A1_temp[[i for i in range(self.n)], [i for i in range(self.n)]] = self.SA1 # embeds self.SA1 vector along the main diagonal.
                SA1 = -2*self.g*self.gb*self.gc*torch.eye(self.n) - self.ga1*A1mask*F.tanh(A1_temp**2) - (10e-5)*torch.eye(self.n)
            elif self.k>1 and self.k<=self.n:
                SA1 = self.SA1 # determined by r_lambda() in initialisation
            else:
                raise ValueError(f"Invalid choice of k.")
            A = 0.5*md([UA1, SA1, torch.transpose(UA1,0,1)]) + 0.5*md([UA2, SA2, torch.transpose(UA2,0,1)])
        else:
            A = self.A # Pass in A matrix

        # Loop through pixels in an image. inputs[:,i,:] has shape [batch_size, input_size] and inputs has shape [batch_size, pixels, input_size]
        for i in range(inputs.shape[1]):
            if self.v1_ind==True and self.v2_ind==False:
                V1 = self.V1_in(inputs[:,i,:]) # V1 has shape [batch size, n]
                Y = torch.matmul(X, torch.transpose(C,0,1)) # Y has shape [batch size, m]
                X = X + self.delta*(torch.matmul(X, torch.transpose(A,0,1)) + torch.matmul(F.relu(Y), torch.transpose(B,0,1)) + V1) # X has shape [batch size, n]

            elif self.v1_ind==False and self.v2_ind==True:
                V2 = self.V2_in(inputs[:,i,:])
                Y = torch.matmul(X, torch.transpose(C,0,1)) + V2 # Y has shape [batch size, m]
                X = X + self.delta*(torch.matmul(X, torch.transpose(A,0,1)) + torch.matmul(F.relu(Y), torch.transpose(B,0,1))) # X has shape [batch size, n]

            elif self.v1_ind==True and self.v2_ind==True:
                V1 = self.V1_in(inputs[:,i,:])
                V2 = self.V2_in(inputs[:,i,:]) # V2 has shape [batch size, m]
                Y = torch.matmul(X, torch.transpose(C,0,1)) + V2 # Y has shape [batch size, m]
                X = X + self.delta*(torch.matmul(X, torch.transpose(A,0,1)) + torch.matmul(F.relu(Y), torch.transpose(B,0,1)) + V1) # X has shape [batch size, n]

            else:
                raise ValueError(f"LurieNet has no inputs!")

        return self.out(X)
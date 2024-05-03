def getData(name:str, dir:str, device:torch.device, r_seed:int, train_bs=64, test_bs=1024):
    """ Function for loading, preprocessing and splitting data for a specified task. Note that LurieNet requires
        tensor containing inputs of shape (N,input_size) where N indicates the batch size.
        args:
            name: String specifying the task. Accepted arguments are scifar10, smnist, psmnist.
             dir: String specifying the directory where the dataset should be saved.
          device: Parameter indicating the type of hardware in use.
          r_seed: Random seed for random generator.
        train_bs: Integer stating the batch size during training (default 64).
         test_bs: Integer stating the batch size during testing (default 1024).
     returns: train, validation and test set dataloaders.
    """

    # Same transformations as RNNs of RNNs (Kozachkov, et al. 2022).
    if name == 'scifar10':
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
        # Returns (image, target) tuple of full train set and test set, where target is index of the target class.
        # Note that if dataset are already downloaded, they are not downloaded again.
        training_data = datasets.CIFAR10(root=dir, train=True, download=True, transform=transform_train)
        testing_data = datasets.CIFAR10(root=dir, train=False, download=True, transform=transform_test)

        # Same settings as RNNs of RNNs (Kozachkov, et al. 2022).
        offset = 2000
        rng = np.random.RandomState(r_seed)
        R = rng.permutation(len(training_data))
        lengths = (len(training_data) - offset, offset)

        # splits full dataset into train and validation set. Overwrites training_data to save memory.
        training_data, val_data = [Subset(training_data, R[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

        # Specifying generator for reproducbiiltiy.
        generator = torch.Generator(device=device)
        generator.manual_seed(r_seed+1)

        # Creating dataloaders - iterables that pass samples in minibatches and reshuffle data every epoch (if shuffle=True).
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=train_bs, shuffle=True, generator=generator)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=test_bs, shuffle=False, generator=generator)
        test_loader = torch.utils.data.DataLoader(testing_data, batch_size=test_bs, shuffle=False, generator=generator)

    if name == 'smnist':

        training_data = datasets.MNIST(dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
        testing_data = datasets.MNIST(dir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))

        # Same settings as RNNs of RNNs (Kozachkov, et al. 2022).
        offset = 2000
        rng = np.random.RandomState(r_seed+2)
        R = rng.permutation(len(training_data))
        lengths = (len(training_data) - offset, offset)

        # splits full dataset into train and validation set. Overwrites training_data to save memory.
        training_data, val_data = [Subset(training_data, R[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

        # Specifying generator for reproducbiiltiy.
        generator = torch.Generator(device=device)
        generator.manual_seed(r_seed+3)

        train_loader = torch.utils.data.DataLoader(training_data, batch_size=train_bs, shuffle=True, generator=generator)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=test_bs, shuffle=False, generator=generator)
        test_loader = torch.utils.data.DataLoader(testing_data,batch_size=test_bs, shuffle=False, generator=generator)

    if name == 'psmnist':

        training_data = datasets.MNIST(root=dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
        testing_data = datasets.MNIST(root=dir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))

        # splitting data into features (x) and targets (y)
        x_train = training_data.train_data
        y_train = training_data.targets
        x_test = testing_data.test_data
        y_test = testing_data.targets

        # permutting input data - same settings as RNNs of RNNs (Kozachkov, et al. 2022).
        torch.manual_seed(r_seed+4)
        perm = torch.randperm(x_train.shape[1]*x_train.shape[2])
        x_train_permuted = x_train.reshape(x_train.shape[0],-1)
        x_train_permuted = x_train_permuted[:, perm]
        x_train_permuted = x_train_permuted.reshape(x_train.shape[0], 28, 28)
        x_test_permuted = x_test.reshape(x_test.shape[0],-1)
        x_test_permuted = x_test_permuted[:, perm]
        x_test_permuted = x_test_permuted.reshape(x_test.shape[0], 28, 28)
        x_train_permuted = add_channels(x_train_permuted)
        x_test_permuted = add_channels(x_test_permuted)

        # Forming the psmnist datasets. Overwrites training_data and testing_data to save memory.
        training_data = torch.utils.data.TensorDataset(x_train_permuted.float(), y_train)
        testing_data = torch.utils.data.TensorDataset(x_test_permuted.float(), y_test)

        # Same settings as RNNs of RNNs (Kozachkov, et al. 2022).
        offset = 2000
        rng = np.random.RandomState(r_seed+5)
        R = rng.permutation(len(training_data))
        lengths = (len(training_data) - offset, offset)

        # splits full dataset into train and validation set. Overwrites training_data to save memory.
        training_data, val_data = [Subset(training_data, R[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]

        # Specifying generator for reproducbiiltiy.
        generator = torch.Generator(device=device)
        generator.manual_seed(r_seed+6)

        # creating dataloaders
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=train_bs, shuffle=True, generator=generator)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=test_bs, shuffle=False, generator=generator)
        test_loader = torch.utils.data.DataLoader(testing_data,batch_size=test_bs, shuffle=False, generator=generator)

    return train_loader, val_loader, test_loader
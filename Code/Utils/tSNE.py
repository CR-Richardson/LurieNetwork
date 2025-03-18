import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms
import sys


root = ''
sys.path.append(root + 'Code/Models')

import CNN
import LurieNet as LN
import kLurieNet_nonHurwitz as LN_k
import Decoder as dec



def import_FMNIST():
    """
    Import fashion MNIST test dataset and create dataloader.
    """

    # zero-center and normalize the distribution of the image tile content
    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for testing
    test_set = datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    # Create data loader, batches have shape (1, 1, 28, 28)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=True)

    # Class labels
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    # Report split sizes
    print( 'Test set has {} instances'.format( len(test_set) ) )

    return test_loader, classes



def get_bestmodel(model_dir, model_name, encoder_bool, n_classes, device):
    """
    Load and return best model determined by model_name, exp_no, and encoder_bool.
    """

    # find and load best model.
    stats = pd.read_csv(model_dir + 'stats.csv')

    # find best model.
    best_model_idx = np.argmax(stats['mean test acc']) + 1
    best_acc = np.max(stats['mean test acc'])
    best_model_dir = model_dir + 'epoch{:03d}.pt'.format(best_model_idx)

    # load best model, state dictionaries, and set to eval mode.
    best_model = torch.load(best_model_dir, map_location=device, weights_only=True)

    tmax = 100
    step = 0.01 
    A_type = 'sym'
    B = None 
    C = None
    bx = None
    by = None

    if encoder_bool == True:
        n = 576
        encoder = CNN.CNN().to(device)
        encoder.load_state_dict( best_model['enc_state_dict'] )
        encoder.eval()
    else:
        n = 784

    if model_name == 'LurieNet':
        print('LurieNet chosen as model!')
        model = LN.LurieNet(n, tmax, step, A_type, B, C, bx, by).to(device)
    elif model_name == 'LurieNet_k':
        print('LurieNet_k chosen as model!')
        g = 1
        k = 3
        model = LN_k.LurieNet_k(n, tmax, step, g, k, A_type, B, C, bx, by).to(device)

    model.load_state_dict( best_model['model_state_dict'], strict=False ) # Include strict=False if problems with loading
    model.eval()

    decoder = dec.Decoder(n, n_classes).to(device)
    decoder.load_state_dict( best_model['dec_state_dict'] )
    decoder.eval()

    print( 'Best model was {} with an accuracy of {}'.format(best_model_idx, best_acc) )

    if encoder_bool == True:
        return encoder, model, decoder
    else:
        return model, decoder
    


def main():

    # User settings: model name and experiment number.
    root = ''
    model_name = 'LurieNet' # LurieNet, LurieNet_k
    exp_no = 1
    encoder_bool = False # build encoder if True
    model_dir = root + 'Models/FMNIST/' + model_name + '/Exp_{:03d}/'.format(exp_no)

    # Hardware agnostic settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in use is: ", device)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load FMNIST dataset
    test_loader, classes = import_FMNIST()
    n_classes = len(classes)

    # Load best model
    if encoder_bool:
        encoder, model, decoder = get_bestmodel(model_dir, model_name, encoder_bool, n_classes, device)
    else:
        model, decoder = get_bestmodel(model_dir, model_name, encoder_bool, n_classes, device)

    print('data loaded and model instantiated!')

    for data, target in test_loader:

        # Send the data and label to the device
        data = data.double().to(device) # (bs, 1, d, d)
        target = target.to(device)

        # pass batch of images through model or encoder / model and store output.
        if not encoder_bool:
            data = data.flatten(start_dim=2, end_dim=3).squeeze(1).to(device) # shape = (bs, d * d)
            output = model(data) # shape = (bs, d * d)
        else:
            output = encoder(data)
            output = model(output) # shape = (bs, d * d)


    print('Output of model computed, reshaping data!')

    # reshape data
    data = data.squeeze(dim=1)
    data = data.flatten(start_dim=1, end_dim=2) # (bs, d * d)

    # Apply t-SNE to output.
    print('starting t-SNE!')
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    output_tsne = tsne.fit_transform( output.numpy() )

    print('t-SNE completed, plotting data!')

    # Plot the result
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    scatter = ax.scatter(output_tsne[:, 0], output_tsne[:, 1], c=target.numpy().astype(int), cmap='tab10', s=1)
    handles, labels = scatter.legend_elements()
    legend = ax.legend(handles, classes, loc="upper left", ncol=len(classes))
    ax.add_artist(legend)
    fig.tight_layout()
    plt.savefig(model_dir + 'tSNE.png')
    plt.show()

    return 0



if __name__ == "__main__":
    main()
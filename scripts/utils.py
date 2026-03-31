"""
Created on Tue Jul 12 22:22:45 2024

@author: djy41
"""
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
import random

def train_epoch(Encoder, DecoderC, DecoderP, device, dataloader, loss_fn, optimizer1, optimizer2, optimizer4):
    """The training loop of autoencoder"""
    #cae.train()#---Set train mode for both the encoder and the decoder
    Encoder.train()
    DecoderC.train()
    DecoderP.train()
    
    train_loss = []
    for _, (a_batch, p_batch, n_batch, y_batch) in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)): #--- This is for triplet batches      
        #---Move tensor to the proper device
        a_batch = a_batch.to(device)
        p_batch = p_batch.to(device)
        n_batch = n_batch.to(device)
        
        z_a_batchC, z_a_batchP = Encoder(a_batch)
        z_p_batchC, z_p_batchP = Encoder(p_batch)
        z_n_batchC, z_n_batchP = Encoder(n_batch)
   
        decoded_a_batch = DecoderC(z_a_batchC) + DecoderP(z_a_batchP)
        decoded_p_batch = DecoderC(z_p_batchC) + DecoderP(z_p_batchP)
        decoded_n_batch = DecoderC(z_n_batchC) + DecoderP(z_n_batchP)        

        #---Evaluate loss
        loss = loss_fn(a_batch, p_batch, n_batch, decoded_a_batch, decoded_p_batch, decoded_n_batch, z_a_batchC, z_p_batchC, z_n_batchC)

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer4.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer4.step()

        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss), z_a_batchC, z_a_batchP
#************************************************************************

def test_epoch(Encoder, DecoderC, DecoderP, device, dataloader, loss_fn):
    """The validation loop of autoencoder on the test dataset"""
    # Set evaluation mode for encoder and decoder
    Encoder.eval()
    DecoderC.eval()
    DecoderP.eval()
    
    with torch.no_grad(): # No need to track the gradients
        #---Define the lists to store the outputs for each batch
        decoded_data = []
        original_data = []
        for x_batch, _, _, _ in dataloader: #---This is for Triplet batches
            # Move tensor to the proper device
            x_batch = x_batch.to(device)
            z1, z2 = Encoder(x_batch) 
            decoded_batch = DecoderC(z1) + DecoderP(z2)
            # Append the network output and the original image to the lists
            decoded_data.append(decoded_batch.cpu())
            original_data.append(x_batch.cpu())
        # Create a single tensor with all the values in the lists
        decoded_data = torch.cat(decoded_data)
        original_data = torch.cat(original_data)
        # Evaluate global loss
        val_loss = loss_fn(decoded_data, original_data)

    return val_loss.data, z1, z2
#************************************************************************

def plot_ae_outputs(Encoder, DecoderC, DecoderP, dataset_opt, epoch, dataset, device, n=10):
    """Saving plot diagrams with reconstructed images in comparision with the original ones for a visual assessment"""
    
    plt.figure(figsize=(16,4.5))
    for i in range(n):

        ax = plt.subplot(2,n,i+1)
        img = dataset[i][0]
        labels = dataset[i][3]
        
        plt.imshow(img.permute((1, 2, 0))) # rgb
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Original images from ' + dataset_opt + ' epoch=' + str(epoch))

        ax = plt.subplot(2, n, i + 1 + n)
        img = img.unsqueeze(0).to(device) # img -> (3, xx, xx) but img.unsqueeze(0) -> (1,3,xx,xx)
        #cae.eval()
        Encoder.eval()
        DecoderC.eval()
        DecoderP.eval()
        
        with torch.no_grad():
            z1, z2 = Encoder(img)
            rec_img = DecoderC(z1) + DecoderP(z2)
        rec_img = rec_img.cpu().squeeze() # rec_img -> (1, 3, xx, xx) but img.squeeze() -> (3,xx,xx)
        plt.imshow(rec_img.permute((1, 2, 0))) # rgb
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Reconstructed images from ' + dataset_opt + ' epoch=' + str(epoch))

    if not os.path.isdir('output'):
        os.mkdir('output')
    # plt.show()
    plt.savefig(f'output/{epoch}_epoch_from_{dataset_opt}.png')
    
#************************************************************************

def checkpoint(model, epoch, val_loss, filename):
    """Saving the model at a specific state"""
    torch.save(model.state_dict(), filename)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            }, filename)
#************************************************************************

def resume(model, filename):
    """Load the trained autoencoder model"""
    checkpoint = torch.load(filename)
    model = model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    return model, epoch, loss
#************************************************************************


#*****************************************************************************
#--- Evaluate Critiron
#*****************************************************************************
def cluster_acc(y_true, y_pred, NUM_CLUSTERS):
    count_matrix = np.zeros((NUM_CLUSTERS, NUM_CLUSTERS), dtype=np.int64)
    for i in range(len(y_pred)):
        count_matrix[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    accuracy = count_matrix[row_ind, col_ind].sum() / len(y_pred)
    return accuracy
#----------------------------------------------------------------------------

#*****************************************************************************
#--- Evaluate Critiron
#*****************************************************************************
def cluster_acc2(y_true, y_pred):
    # Convert to NumPy arrays
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.array(y_pred)

    # Get unique labels and map to contiguous indices
    labels_true = np.unique(y_true_np)
    labels_pred = np.unique(y_pred_np)

    # Create mapping arrays for vectorized indexing
    true_map = {label: i for i, label in enumerate(labels_true)}
    pred_map = {label: i for i, label in enumerate(labels_pred)}

    # Vectorized mapping using np.vectorize
    map_true_vec = np.vectorize(lambda x: true_map[int(x)])
    map_pred_vec = np.vectorize(lambda x: pred_map[int(x)])

    y_true_idx = map_true_vec(y_true_np)
    y_pred_idx = map_pred_vec(y_pred_np)

    # Build contingency matrix
    NUM_CLUSTERS = max(len(labels_true), len(labels_pred))
    count_matrix = np.zeros((NUM_CLUSTERS, NUM_CLUSTERS), dtype=np.int64)
    np.add.at(count_matrix, (y_pred_idx, y_true_idx), 1)

    # Hungarian algorithm to maximize total matches
    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    accuracy = count_matrix[row_ind, col_ind].sum() / len(y_true_np)

    return accuracy

#*****************************************************************************
#--- Evaluate Critiron
#*****************************************************************************
def calculate_purity(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster_index in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster_index], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster_index] = winner

    return accuracy_score(y_true, y_voted_labels)
#----------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Optional: makes more ops deterministic (can raise errors if an op has no deterministic impl)
        # torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
#----------------------------------------------------------------------------
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path

# load saved Xtrain
X_train = np.load('xtrain_111312.npy')

# load the mask
mask_file = os.path.join(data_path, '/Diffusion/nodif_brain_mask.nii.gz')
mask_image = nib.load(mask_file)
# Return floating point image data with necessary scaling applied
mask = mask_image.get_fdata()

# reshape mask
mask_vox = np.reshape(mask,(3658350))

 #find out where Xtrain is 0
np.size(np.where(np.any(X_train==0 , axis=1)))

# make new xtrain to get rid of the 0s
mask_vox_new = np.ones(np.shape(X_train)[0])
mask_vox_new[np.any(X_train==0 , axis=1)] = 0
X_train_new = X_train[mask_vox_new == 1,:]

print(np.sum(mask_vox_new))
print(np.sum(mask_vox))
# as you can see xtrain size has been reduced to get rid of the 0s

# define msdki function
def msdki(b,D,K):
  return np.exp(-b*D + 1/6 * (b**2) * (D**2) * K)
  #return np.exp(-b*D)
  #where $D$ and $K$ are the mean signal diffusivity and mean signal kurtosis

# load and define b-values
b_values_file = os.path.join(data_path, '/Diffusion/bvals')
b_values = np.loadtxt(b_values_file)
# make b_values smaller
b_values = b_values*10**-3

# create the neural network class
np.random.seed(42) #set random seed 
class Net(nn.Module):
    def __init__(self, b_values_no0):
        super(Net, self).__init__()

        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        for i in range(3): # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 2))

    def forward(self, X):
        params = torch.abs(self.encoder(X)) # D, K
        D = params[:, 0].unsqueeze(1)
        K = params[:, 1].unsqueeze(1)

        
        D = torch.clamp(D, min=0, max=3)
        K = torch.clamp(K, min=0, max=2)
        
        #np.exp(-b*D + 1/6 * b**2 * D**2 * K)
        X = torch.exp(-self.b_values_no0*D + 1/6 * self.b_values_no0**2 * D**2 * K)
        

        return X, D, K
 
# normalising the Xtrain
X_train1 = X_train_new + np.finfo(float).eps
X_train_norm = X_train1[:,b_values == 0.005] #all b0s
b_values_mean = np.mean(X_train_norm , axis = 1)
#mean value of b0 in each voxel
b_values_mean_total = np.tile(b_values_mean,(288,1)).T
X_train_norm = X_train1/b_values_mean_total

# direction averaging
X_train_norm_da = np.zeros((np.shape(X_train_norm)[0],4))
X_train_b0 = X_train_norm[:,b_values == 0.005] #all b0s
X_train_meanb0=np.array([np.mean(X_train_b0 , axis = 1)])
X_train_norm_da[:,0] = X_train_meanb0
X_train_b1 = X_train_norm[:, np.round_(b_values) == 1.000 ]
X_train_meanb1 = np.array([np.mean(X_train_b1 , axis = 1)])
X_train_norm_da[:,1] = X_train_meanb1
X_train_b2 = X_train_norm[:, np.round_(b_values) == 2.000 ]
X_train_meanb2 = np.array([np.mean(X_train_b2 , axis = 1)])
X_train_norm_da[:,2] = X_train_meanb2
X_train_b3 = X_train_norm[:, np.round_(b_values) == 3.000 ]
X_train_meanb3 = np.array([np.mean(X_train_b3 , axis = 1)])
X_train_norm_da[:,3] = X_train_meanb3

# set new b-values
b_values_da = np.array([0,1,2,3])

# initiate Network
b_values_no0 = torch.FloatTensor(b_values_da)
net = Net(b_values_no0)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0001)

batch_size = 128
num_batches = len(X_train_norm_da) // batch_size
trainloader = utils.DataLoader(torch.from_numpy(X_train_norm_da.astype(np.float32)),
                                batch_size = batch_size, 
                                shuffle = True,
                                num_workers = 2,
                                drop_last = True)
                                
                                
# Best loss
best = 1e16
num_bad_epochs = 0
patience = 10

# Train
for epoch in range(1000): 
    print("-----------------------------------------------------------------")
    print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
    net.train()
    running_loss = 0.

    for i, X_batch in enumerate(tqdm(trainloader), 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        X_pred, D_pred, K_pred = net(X_batch)
        loss = criterion(X_pred, X_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
      
    print("Loss: {}".format(running_loss))
    # early stopping
    if running_loss < best:
        print("############### Saving good model ###############################")
        final_model = net.state_dict()
        best = running_loss
        num_bad_epochs = 0
    else:
        num_bad_epochs = num_bad_epochs + 1
        if num_bad_epochs == patience:
            print("Done, best loss: {}".format(best))
            break
print("Done")
# Restore best model
net.load_state_dict(final_model)

# get the predictions
net.eval()
with torch.no_grad():
    X_pred, D, K = net(torch.from_numpy(X_train_norm_da.astype(np.float32)))
    
D = D.numpy()
K = K.numpy()
X_pred = X_pred.numpy()

# reshape D
d681 = np.zeros(np.shape(mask_vox_new))
d681[np.where(mask_vox_new==1)] = np.squeeze(D)
d =np.zeros(np.shape(mask_vox))
d[np.where(mask_vox==1)] = d681
d_map = np.reshape(d,(np.shape(mask)))

# reshape K
k681 = np.zeros(np.shape(mask_vox_new))
k681[np.where(mask_vox_new==1)] = np.squeeze(K)
k =np.zeros(np.shape(mask_vox))
k[np.where(mask_vox==1)] = k681
k_map = np.reshape(k,(np.shape(mask)))

# save d_map and k_map
d_map_img = nib.Nifti1Image(d_map, np.eye(4))
nib.save(d_map_img,'d_map_img_111312.nii.gz')

k_map_img = nib.Nifti1Image(k_map, np.eye(4))
nib.save(k_map_img,'k_map_img_111312.nii.gz')

# plot to visualise the maps
fig, ax = plt.subplots(1, 2, figsize=(20,20))

D_plot = ax[0].imshow(d_map[:, :, 50].T, origin = 'lower', cmap='gray', clim=(0,2))
ax[0].set_title('D, estimated from self-supervised machine learning')
ax[0].set_xticks([])
ax[0].set_yticks([])
fig.colorbar(D_plot, ax=ax[0], fraction=0.046, pad=0.04)

K_plot = ax[1].imshow(k_map[:, :, 50].T, origin = 'lower', cmap='gray', clim=(0,3))
ax[1].set_title('K, estimated from self-supervised machine learning')
ax[1].set_xticks([])
ax[1].set_yticks([])
fig.colorbar(K_plot, ax=ax[1],fraction=0.046, pad=0.04)

# import libraries
import numpy as np
import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path

# load image voxels
file = os.path.join(data_path, '/Users/kimia/Downloads/Diffusion/data.nii.gz')
image = nib.load(file)
data_hcp = image.get_fdata()

# load mask voxles
fl = os.path.join(data_path, '/Users/kimia/Downloads/Diffusion/nodif_brain_mask.nii.gz')
img = nib.load(fl)
mask = img.get_fdata()

# reshape image and mask in the form of a X-train
data_vox = np.reshape(data_hcp,(3658350,288))
mask_vox = np.reshape(mask,(3658350))

# pick out the voxels in image that correspond to mask (so only brain voxles are picked out)
X_train= data_vox[mask_vox==1,:]

# save X_train
np.save('xtrain_111312', X_train)

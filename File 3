# import libraries

# Reconstruction module
import dipy.reconst.msdki as msdki

from dipy.core.gradients import gradient_table

import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path

# load b-values and b-vestors of the image
fl1_111312 = os.path.join(data_path, '/Users/kimia/Downloads/Diffusion/bvals')
bvals_111312 = np.loadtxt(fl1_111312)

fl2_111312 = os.path.join(data_path, '/Users/kimia/Downloads/Diffusion/bvecs')
bvecs_111312 = np.loadtxt(fl2_111312)

# make gardient table
gtab_111312 = gradient_table(bvals_111312, bvecs_111312)

# load image
dfile_111312 = os.path.join(data_path, '/Users/kimia/Downloads/Diffusion/data.nii.gz')
dimg_111312 = nib.load(dfile_111312)
image_111312 = dimg_111312.get_fdata()

# load mask
mfile_111312 = os.path.join(data_path, '/Users/kimia/Downloads/Diffusion/nodif_brain_mask.nii.gz')
mimg_111312 = nib.load(mfile_111312)
mask_111312 = mimg_111312.get_fdata()

# fit and save the data
msdki_model_111312 = msdki.MeanDiffusionKurtosisModel(gtab_111312)
msdki_fit_111312 = msdki_model_111312.fit(image_111312 ,mask_111312)

MSD_111312 = msdki_fit_111312.msd
MSK_111312 = msdki_fit_111312.msk

MSD_111312_img = nib.Nifti1Image(MSD_111312, np.eye(4))
nib.save(MSD_111312_img,'MSD_111312_img.nii.gz')
MSK_111312_img = nib.Nifti1Image(MSK_111312, np.eye(4))
nib.save(MSK_111312_img,'MSK_111312_img.nii.gz')

# import libraries
import nibabel as nib
import os
from nibabel.testing import data_path
import statistics
import math
import numpy as np

# load the d_maps and k_maps of scan with suoervised learning and LSQ
# clip the MSD and MSK (the mapsfrom LSQ) to be within certain theresholds

d_map_img_111312 = nib.load('d_map_img_111312.nii.gz').get_fdata()
k_map_img_111312 = nib.load('k_map_img_111312.nii.gz').get_fdata()
MSD_111312_img = nib.load('MSD_111312_img.nii.gz').get_fdata()
MSD_111312_img[MSD_111312_img>3] = 3
MSD_111312_img[MSD_111312_img<0] = 0
MSK_111312_img = nib.load('MSK_111312_img.nii.gz').get_fdata()
MSK_111312_img[MSK_111312_img>2] = 2
MSK_111312_img[MSK_111312_img<0] = 0


# To visualise the maps, matplotlib was used( the code is in file 5)


# define contrast and contrast-to-noise ratio
def contrast(a,b):
    return np.abs(a-b)/(0.5 * (a+b))

def contrast_to_noise_ratio(a,b,c,d):
    return np.abs(a-b)/math.sqrt(c+d)

# load ADC and FA
a_111312 = nib.load('/Users/kimia/Downloads/Diffusion/adc.nii.gz').get_fdata()
f_111312 = nib.load('/Users/kimia/Downloads/Diffusion/fa.nii.gz').get_fdata()

# load mask
mfile_111312 = os.path.join(data_path, '/Users/kimia/Downloads/Diffusion/nodif_brain_mask.nii.gz')
mimg_111312 = nib.load(mfile_111312)
mask_111312 = mimg_111312.get_fdata()

# make grey matter nad white matter masks
wm_mask_111312 = (a_111312 < 0.002) & (f_111312 > 0.3) & (mask_111312 == 1)
gm_mask_111312 = (a_111312 < 0.002) & (f_111312 < 0.3) & (mask_111312 == 1)

# pick out voxels that correspond to each mask
d_map_img_111312_wm = d_map_img_111312[wm_mask_111312 == 1]
d_map_img_111312_gm = d_map_img_111312[gm_mask_111312 == 1]

MSD_111312_img_wm = MSD_111312_img[wm_mask_111312 == 1]
MSD_111312_img_gm = MSD_111312_img[gm_mask_111312 == 1]

k_map_img_111312_wm = k_map_img_111312[wm_mask_111312 == 1]
k_map_img_111312_gm = k_map_img_111312[gm_mask_111312 == 1]

MSK_111312_img_wm = MSK_111312_img[wm_mask_111312 == 1]
MSK_111312_img_gm = MSK_111312_img[gm_mask_111312 == 1]

#calculate mean and standard deviation for each of the 4 maps
mu_d_map_img_111312_wm = statistics.mean(d_map_img_111312_wm)
mu_d_map_img_111312_gm = statistics.mean(d_map_img_111312_gm)
C_d_map_111312 = contrast(mu_d_map_img_111312_wm,mu_d_map_img_111312_gm)
v_d_map_img_111312_wm = statistics.pvariance(d_map_img_111312_wm)
v_d_map_img_111312_gm = statistics.pvariance(d_map_img_111312_gm)
CR_d_map_111312 = contrast_to_noise_ratio(mu_d_map_img_111312_wm,mu_d_map_img_111312_gm,v_d_map_img_111312_wm,v_d_map_img_111312_gm)

mu_MSD_111312_img_wm = statistics.mean(MSD_111312_img_wm)
mu_MSD_111312_img_gm = statistics.mean(MSD_111312_img_gm)
C_MSD_111312 = contrast(mu_MSD_111312_img_wm,mu_MSD_111312_img_gm)
v_MSD_111312_img_wm = statistics.pvariance(MSD_111312_img_wm)
v_MSD_111312_img_gm = statistics.pvariance(MSD_111312_img_gm)
CR_MSD_111312 = contrast_to_noise_ratio(mu_MSD_111312_img_wm,mu_MSD_111312_img_gm,v_MSD_111312_img_wm,v_MSD_111312_img_gm)

mu_k_map_img_111312_wm = statistics.mean(k_map_img_111312_wm)
mu_k_map_img_111312_gm = statistics.mean(k_map_img_111312_gm)
C_k_map_111312 = contrast(mu_k_map_img_111312_wm,mu_k_map_img_111312_gm)
v_k_map_img_111312_wm = statistics.pvariance(k_map_img_111312_wm)
v_k_map_img_111312_gm = statistics.pvariance(k_map_img_111312_gm)
CR_k_map_111312 = contrast_to_noise_ratio_d_map_111312 = contrast_to_noise_ratio(mu_k_map_img_111312_wm,mu_k_map_img_111312_gm,v_k_map_img_111312_wm,v_k_map_img_111312_gm)

mu_MSK_111312_img_wm = statistics.mean(MSK_111312_img_wm)
mu_MSK_111312_img_gm = statistics.mean(MSK_111312_img_gm)
C_MSK_111312 = contrast(mu_MSK_111312_img_wm, mu_MSK_111312_img_gm)
v_MSK_111312_img_wm = statistics.pvariance(MSK_111312_img_wm)
v_MSK_111312_img_gm = statistics.pvariance(MSK_111312_img_gm)
CR_MSK_111312 = contrast_to_noise_ratio(mu_MSK_111312_img_wm,mu_MSK_111312_img_gm,v_MSK_111312_img_wm,v_MSK_111312_img_gm)

# to compare conrast and contrast-to-noise ratio bar plots were plotted using pandas (the code is in file 5)

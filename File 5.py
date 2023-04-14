# note : the visualisations here are for all 6 images 

# import libraries
import matplotlib.pyplot as plt
import pandas as pd


# the maps in saggital view
fig, ax = plt.subplots(6, 4, figsize=(50,100),subplot_kw={'xticks': [], 'yticks': []})


ss = 60


#103818

D_plot0 = ax[0,2].imshow(d_map_img_103818[ss,:,:].T, origin='lower', clim=(0,2) , cmap='magma')
ax[0,2].set_title('D [$\mu m^2$$ms^{-1}$], self-supervised' , fontsize= 40 ,x=0.5, y=1.2)
cbar = fig.colorbar(D_plot0, ax=ax[0,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)

K_plot0 = ax[0,3].imshow(k_map_img_103818[ss,:,:].T, origin='lower', clim=(0,2))
ax[0,3].set_title('K, self-supervised' ,fontsize= 40 ,x=0.5, y=1.2)
cbar = fig.colorbar(K_plot0, ax=ax[0,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot0 = ax[0,0].imshow(MSD_103818_img[ss,:,:].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[0,0].set_title('D [$\mu m^2$$ms^{-1}$], LSQ' ,fontsize= 40 ,x=0.5, y=1.2)
ax[0,0].set_ylabel('scan1' , rotation="horizontal" , fontsize=50, color='blue')
ax[0,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot0, ax=ax[0,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot0 = ax[0,1].imshow(MSK_103818_img[ss,:,:].T, origin='lower', clim=(0,2))
ax[0,1].set_title('K, LSQ' ,fontsize=40 ,x=0.5, y=1.2)
cbar = fig.colorbar(K_t_plot0, ax=ax[0,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)

#111312

D_plot1 = ax[1,2].imshow(d_map_img_111312[ss,:,:].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot1, ax=ax[1,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot1 = ax[1,3].imshow(k_map_img_111312[ss,:,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot1, ax=ax[1,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot1 = ax[1,0].imshow(MSD_111312_img[ss,:,:].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[1,0].set_ylabel('scan2' , rotation="horizontal" , fontsize=50, color='blue')
ax[1,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot1, ax=ax[1,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot1 = ax[1,1].imshow(MSK_111312_img[ss,:,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot1, ax=ax[1,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

#200109

D_plot2 = ax[2,2].imshow(d_map_img_200109[ss,:,:].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot2, ax=ax[2,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot2 = ax[2,3].imshow(k_map_img_200109[ss,:,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot2, ax=ax[2,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot2 = ax[2,0].imshow(MSD_200109_img[ss,:,:].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[2,0].set_ylabel('scan3' , rotation="horizontal" , fontsize=50, color='blue')
ax[2,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot2, ax=ax[2,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot2 = ax[2,1].imshow(MSK_200109_img[ss,:,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot2, ax=ax[2,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

#200614

D_plot3 = ax[3,2].imshow(d_map_img_200614[ss,:,:].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot3, ax=ax[3,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot3 = ax[3,3].imshow(k_map_img_200614[ss,:,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot3, ax=ax[3,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot3 = ax[3,0].imshow(MSD_200614_img[ss,:,:].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[3,0].set_ylabel('scan4' , rotation="horizontal" , fontsize=50, color='blue')
ax[3,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot3, ax=ax[3,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot3 = ax[3,1].imshow(MSK_200614_img[ss,:,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot3, ax=ax[3,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

#204521

D_plot4 = ax[4,2].imshow(d_map_img_204521[ss,:,:].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot4, ax=ax[4,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot4 = ax[4,3].imshow(k_map_img_204521[ss,:,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot4, ax=ax[4,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot4 = ax[4,0].imshow(MSD_204521_img[ss,:,:].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[4,0].set_ylabel('scan5' , rotation="horizontal" , fontsize=50, color='blue')
ax[4,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot4, ax=ax[4,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot4 = ax[4,1].imshow(MSK_204521_img[ss,:,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot4, ax=ax[4,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

#250427

D_plot5 = ax[5,2].imshow(d_map_img_250427[ss,:,:].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot5, ax=ax[5,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot5 = ax[5,3].imshow(k_map_img_250427[ss,:,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot5, ax=ax[5,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot5 = ax[5,0].imshow(MSD_250427_img[ss,:,:].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[5,0].set_ylabel('scan6' , rotation="horizontal" , fontsize=50, color='blue')
ax[5,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot5, ax=ax[5,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot5 = ax[5,1].imshow(MSK_250427_img[ss,:,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot5, ax=ax[5,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

plt.subplots_adjust(hspace=-0.75)
plt.show()

# Grey matter and white matter masks for one of the scans in saggital view
plt.imshow(wm_mask_103818[ss,:,:].T , origin='lower')
plt.imshow(gm_mask_103818[ss,:,:].T , origin='lower')

# the maps in axial view
fig, ax = plt.subplots(6, 4, figsize=(50,100),subplot_kw={'xticks': [], 'yticks': []})


As = 60

#103818

D_plot0 = ax[0,2].imshow(d_map_img_103818[:,:,As].T, origin='lower', clim=(0,2) , cmap='magma')
ax[0,2].set_title('D [$\mu m^2$$ms^{-1}$], self-supervised' , fontsize= 40 ,x=0.5, y=1.2)
cbar = fig.colorbar(D_plot0, ax=ax[0,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)

K_plot0 = ax[0,3].imshow(k_map_img_103818[:,:,As].T, origin='lower', clim=(0,2))
ax[0,3].set_title('K, self-supervised' ,fontsize= 40 ,x=0.5, y=1.2)
cbar = fig.colorbar(K_plot0, ax=ax[0,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot0 = ax[0,0].imshow(MSD_103818_img[:,:,As].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[0,0].set_title('D [$\mu m^2$$ms^{-1}$], LSQ' ,fontsize= 40 ,x=0.5, y=1.2)
ax[0,0].set_ylabel('scan1' , rotation="horizontal" , fontsize=50, color='blue')
ax[0,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot0, ax=ax[0,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot0 = ax[0,1].imshow(MSK_103818_img[:,:,As].T, origin='lower', clim=(0,2))
ax[0,1].set_title('K, LSQ' ,fontsize=40 ,x=0.5, y=1.2)
cbar = fig.colorbar(K_t_plot0, ax=ax[0,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)

#111312

D_plot1 = ax[1,2].imshow(d_map_img_111312[:,:,As].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot1, ax=ax[1,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot1 = ax[1,3].imshow(k_map_img_111312[:,:,As].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot1, ax=ax[1,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot1 = ax[1,0].imshow(MSD_111312_img[:,:,As].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[1,0].set_ylabel('scan2' , rotation="horizontal" , fontsize=50, color='blue')
ax[1,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot1, ax=ax[1,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot1 = ax[1,1].imshow(MSK_111312_img[:,:,As].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot1, ax=ax[1,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

#200109

D_plot2 = ax[2,2].imshow(d_map_img_200109[:,:,As].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot2, ax=ax[2,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot2 = ax[2,3].imshow(k_map_img_200109[:,:,As].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot2, ax=ax[2,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot2 = ax[2,0].imshow(MSD_200109_img[:,:,As].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[2,0].set_ylabel('scan3' , rotation="horizontal" , fontsize=50, color='blue')
ax[2,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot2, ax=ax[2,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot2 = ax[2,1].imshow(MSK_200109_img[:,:,As].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot2, ax=ax[2,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

#200614

D_plot3 = ax[3,2].imshow(d_map_img_200614[:,:,As].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot3, ax=ax[3,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot3 = ax[3,3].imshow(k_map_img_200614[:,:,As].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot3, ax=ax[3,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot3 = ax[3,0].imshow(MSD_200614_img[:,:,As].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[3,0].set_ylabel('scan4' , rotation="horizontal" , fontsize=50, color='blue')
ax[3,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot3, ax=ax[3,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot3 = ax[3,1].imshow(MSK_200614_img[:,:,As].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot3, ax=ax[3,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

#204521

D_plot4 = ax[4,2].imshow(d_map_img_204521[:,:,As].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot4, ax=ax[4,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot4 = ax[4,3].imshow(k_map_img_204521[:,:,As].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot4, ax=ax[4,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot4 = ax[4,0].imshow(MSD_204521_img[:,:,As].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[4,0].set_ylabel('scan5' , rotation="horizontal" , fontsize=50, color='blue')
ax[4,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot4, ax=ax[4,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot4 = ax[4,1].imshow(MSK_204521_img[:,:,As].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot4, ax=ax[4,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

#250427

D_plot5 = ax[5,2].imshow(d_map_img_250427[:,:,As].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot5, ax=ax[5,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot5 = ax[5,3].imshow(k_map_img_250427[:,:,As].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot5, ax=ax[5,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot5 = ax[5,0].imshow(MSD_250427_img[:,:,As].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[5,0].set_ylabel('scan6' , rotation="horizontal" , fontsize=50, color='blue')
ax[5,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot5, ax=ax[5,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot5 = ax[5,1].imshow(MSK_250427_img[:,:,As].T, origin='lower', clim=(0,2), cmap = 'viridis')
cbar = fig.colorbar(K_t_plot5, ax=ax[5,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

plt.subplots_adjust(hspace=-0.5)
plt.show()


# Grey matter and white matter masks for one of the scans in axial view
plt.imshow(wm_mask_103818[:,:,As].T , origin='lower')
plt.imshow(gm_mask_103818[:,:,As].T , origin='lower')


# the maps in coronal view
fig, ax = plt.subplots(6, 4, figsize=(50,100),subplot_kw={'xticks': [], 'yticks': []})


cs = 87


#103818

D_plot0 = ax[0,2].imshow(d_map_img_103818[:,cs,:].T, origin='lower', clim=(0,2) , cmap='magma')
ax[0,2].set_title('D [$\mu m^2$$ms^{-1}$], self-supervised' , fontsize= 40 ,x=0.5, y=1.2)
cbar = fig.colorbar(D_plot0, ax=ax[0,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)

K_plot0 = ax[0,3].imshow(k_map_img_103818[:,cs,:].T, origin='lower', clim=(0,2))
ax[0,3].set_title('K, self-supervised' ,fontsize= 40 ,x=0.5, y=1.2)
cbar = fig.colorbar(K_plot0, ax=ax[0,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot0 = ax[0,0].imshow(MSD_103818_img[:,cs,:].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[0,0].set_title('D [$\mu m^2$$ms^{-1}$], LSQ' ,fontsize= 40 ,x=0.5, y=1.2)
ax[0,0].set_ylabel('scan1' , rotation="horizontal" , fontsize=50, color='blue')
ax[0,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot0, ax=ax[0,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot0 = ax[0,1].imshow(MSK_103818_img[:,cs,:].T, origin='lower', clim=(0,2))
ax[0,1].set_title('K, LSQ' ,fontsize=40 ,x=0.5, y=1.2)
cbar = fig.colorbar(K_t_plot0, ax=ax[0,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)

#111312

D_plot1 = ax[1,2].imshow(d_map_img_111312[:,cs,:].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot1, ax=ax[1,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot1 = ax[1,3].imshow(k_map_img_111312[:,cs,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot1, ax=ax[1,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot1 = ax[1,0].imshow(MSD_111312_img[:,cs,:].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[1,0].set_ylabel('scan2' , rotation="horizontal" , fontsize=50, color='blue')
ax[1,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot1, ax=ax[1,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot1 = ax[1,1].imshow(MSK_111312_img[:,cs,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot1, ax=ax[1,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

#200109

D_plot2 = ax[2,2].imshow(d_map_img_200109[:,cs,:].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot2, ax=ax[2,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot2 = ax[2,3].imshow(k_map_img_200109[:,cs,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot2, ax=ax[2,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot2 = ax[2,0].imshow(MSD_200109_img[:,cs,:].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[2,0].set_ylabel('scan3' , rotation="horizontal" , fontsize=50, color='blue')
ax[2,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot2, ax=ax[2,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot2 = ax[2,1].imshow(MSK_200109_img[:,cs,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot2, ax=ax[2,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

#200614

D_plot3 = ax[3,2].imshow(d_map_img_200614[:,cs,:].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot3, ax=ax[3,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot3 = ax[3,3].imshow(k_map_img_200614[:,cs,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot3, ax=ax[3,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot3 = ax[3,0].imshow(MSD_200614_img[:,cs,:].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[3,0].set_ylabel('scan4' , rotation="horizontal" , fontsize=50, color='blue')
ax[3,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot3, ax=ax[3,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot3 = ax[3,1].imshow(MSK_200614_img[:,cs,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot3, ax=ax[3,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

#204521

D_plot4 = ax[4,2].imshow(d_map_img_204521[:,cs,:].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot4, ax=ax[4,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot4 = ax[4,3].imshow(k_map_img_204521[:,cs,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot4, ax=ax[4,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot4 = ax[4,0].imshow(MSD_204521_img[:,cs,:].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[4,0].set_ylabel('scan5' , rotation="horizontal" , fontsize=50, color='blue')
ax[4,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot4, ax=ax[4,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot4 = ax[4,1].imshow(MSK_204521_img[:,cs,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot4, ax=ax[4,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

#250427

D_plot5 = ax[5,2].imshow(d_map_img_250427[:,cs,:].T, origin='lower', clim=(0,2) , cmap='magma')
cbar = fig.colorbar(D_plot5, ax=ax[5,2], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_plot5 = ax[5,3].imshow(k_map_img_250427[:,cs,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_plot5, ax=ax[5,3],fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
D_t_plot5 = ax[5,0].imshow(MSD_250427_img[:,cs,:].T * 1000, origin='lower', clim=(0,2) , cmap='magma')
ax[5,0].set_ylabel('scan6' , rotation="horizontal" , fontsize=50, color='blue')
ax[5,0].yaxis.set_label_coords(-.4, .5)
cbar = fig.colorbar(D_t_plot5, ax=ax[5,0], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        
K_t_plot5 = ax[5,1].imshow(MSK_250427_img[:,cs,:].T, origin='lower', clim=(0,2))
cbar = fig.colorbar(K_t_plot5, ax=ax[5,1], fraction=0.046, pad=0.04)
for t in cbar.ax.get_yticklabels():
     t.set_fontsize(20)
        

plt.subplots_adjust(hspace=-0.75)
plt.show()


# Grey matter and white matter masks for one of the scans in coronal view
plt.imshow(wm_mask_103818[:,cs,:].T , origin='lower')
plt.imshow(gm_mask_103818[:,cs,:].T , origin='lower')


#bar plots comparing contrast & contrast to noise ratio between the two methods

# d-map, contrast
plotdata = pd.DataFrame({

    "D, self_supervised":[C_d_map_103818,C_d_map_111312,C_d_map_200109,C_d_map_200614,C_d_map_204521,C_d_map_250427],

    "D, LSQ":[C_MSD_103818,C_MSD_111312,C_MSD_200109,C_MSD_200614,C_MSD_204521,C_MSD_250427]},

    index=["scan1", "scan2", "scan3", "scan4", "scan5", "scan6"])

plotdata.plot(kind="bar",figsize=(15, 8), color =['red', 'purple'])

plt.title("Contrast comparison for diffision")

plt.ylabel("Contrast")



# d-map, contrast-to-noise ratio
plotdata = pd.DataFrame({

    "D, self_supervised":[CR_d_map_103818,CR_d_map_111312,CR_d_map_200109,CR_d_map_200614,CR_d_map_204521,CR_d_map_250427],

    "D, LSQ":[CR_MSD_103818,CR_MSD_111312,CR_MSD_200109,CR_MSD_200614,CR_MSD_204521,CR_MSD_250427]},

    index=["scan1", "scan2", "scan3", "scan4", "scan5", "scan6"])

plotdata.plot(kind="bar",figsize=(15, 8), color =['red', 'purple'])

plt.title("Contrast-to-noise-ratio comparison for diffision")

#plt.xlabel("")

plt.ylabel("Contrast-to-noise-ratio")



# k-map, contrast
plotdata = pd.DataFrame({

    "K, self_supervised":[C_k_map_103818,C_k_map_111312,C_k_map_200109,C_k_map_200614,C_k_map_204521,C_k_map_250427],

    "K, LSQ":[C_MSK_103818,C_MSK_111312,C_MSK_200109,C_MSK_200614,C_MSK_204521,C_MSK_250427]},

    index=["scan1", "scan2", "scan3", "scan4", "scan5", "scan6"])

plotdata.plot(kind="bar",figsize=(15, 8), color =['blue', 'green'])

plt.title("Contrast comparison for kurtosis")

#plt.xlabel("")

plt.ylabel("Contrast")



# k-map, contrast-to-noise ratio
plotdata = pd.DataFrame({

    "K, self_supervised":[CR_k_map_103818,CR_k_map_111312,CR_k_map_200109,CR_k_map_200614,CR_k_map_204521,CR_k_map_250427],

    "K, LSQ":[CR_MSK_103818,CR_MSK_111312,CR_MSK_200109,CR_MSK_200614,CR_MSK_204521,CR_MSK_250427]},

    index=["scan1", "scan2", "scan3", "scan4", "scan5", "scan6"])

plotdata.plot(kind="bar",figsize=(15, 8), color =['blue', 'green'])

plt.title("Contrast-to-noise-ratio comparison for kurtosis")

#plt.xlabel("")

plt.ylabel("Contrast-to-noise-ratio")









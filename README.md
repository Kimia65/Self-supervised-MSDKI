# Self-supervised MSDKI
Self-supervised mean signal kurtosis imaging in the brain (fulfillment of MCM iBSc)


This project is done on 6 HCP images with ID numbers 103818, 111312, 200109, 200614, 204521, 250427 each one corresponding respectively to scan1, 2, 3, 4, 5, 6 in the project paper.

Files 1, 2, 3 and 4 only show the coding done for one of the scans ( 111312 ) but the same codes can be replicated for the other scans as well, you just have to load in the different HCP data.

File 1 shows how to load the HCP data for one of the scans and creat and save the X-train for it which is needed for self supervised learning.

File 2 contains the code for self-supervised learning.

File 3 contains the code for LSQ method.

File 4 shows how to load the d and k maps, create white matter and grey matter masks, calculate contast and contrast-to-noise ratio.

After doing the same process on the other 5 scans, we can visualise the results and so file 5 shows how to plot the results and namely, plots 2-9 in the project paper.

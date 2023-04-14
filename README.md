# Self-supervised MSDKI
Self-supervised mean signal kurtosis imaging in the brain (fulfillment of MCM iBSc)

this project is done on 6 HCP images with code id numbers 103818, 111312, 200109, 200614, 204521, 250427 each one corresponding respectively to scan1,2,3,4,5,6 in the project

files 1, 2, 3 and 4 only show the coding done for one of the scans ( 111312 ) but the same codes can be done dor the other scans as well, you just have to load in the different HCP data

file 1 shows how to load the HCP data for one of the scans and creat and save the X-train for it which is needed for self supervised learning
file 2 contains the code for self-supervised learning
file 3 contains the code for LSQ method
file 4 shows how to load the d and k maps, create white matter and grey matter masks, calculate contast and contrast-to-noise ratio
after doing the same process on the other 5 images, we can visualise the results and so file 5 show how to plot the results and namely, plots 2-9 in the project paper

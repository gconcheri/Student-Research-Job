## TODO:

Sistema ED_Cs_LSites.py -> al momento non sto trovando ground state con h=10**-2.5 e poi facendo simulazione con h=0, sto semplicemente tenendo h=0 tutto il tempo. (da cambiare)

SVD truncation:

1. Fai isometric tensor networks lungo un sweep, e poi fai svd indietro per andare a vedere singular values e in caso fare truncation lungo il sweep

2. Lavora sul caso che hai già fatto, ovvero troncare singular values della two site slice. Idea nuova: fare nuovo sweep con interpolative decomposition, in pratica fare un altro giro di algoritmo, ma con errore più alto, in modo da far sì che solo certi elementi vengano tenuti, numero di singular values automaticamente diventino meno.

3. Convolution
DONE - try to see what shape of convolution function works better. a Gaussian must work.
either use a Chebysehv with 500 bond dimension - and then truncate it with SVD 
or use TCI by giving it all of the values around the center of the gaussian (you can give it more pivot points)
DONE - ~~do it by hand with something that sums up to 1: [0.05, 0.2, 0.5, 0.2, 0.05] for example, and then see how it works. This is a convolution with a kernel of size 5, and the kernel is normalized to sum to 1. You can adjust the values in the kernel to see how it affects the results.~~
- then add other two bits to the initial tci so that convolution is correct on the right interval

- search for convolution filter that works better than Gaussian to convolve well polynomials of 2nd order




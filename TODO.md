## TODO:

(least square optimization with string tension)

----

Sistema ED_Cs_LSites.py -> al momento non sto trovando ground state con h=10**-2.5 e poi facendo simulazione con h=0, sto semplicemente tenendo h=0 tutto il tempo. (da cambiare)

SVD truncation:


1. Fai isometric tensor networks lungo un sweep, e poi fai svd indietro per andare a vedere singular values e in caso fare truncation lungo il sweeo

2. Lavora sul caso che hai già fatto: ovvero troncare singular values della two site slice. Idea nuova: fare nuovo sweep con interpolative decomposition, in pratica fare un altro giro di algoritmo, ma con errore più alto, in modo da far sì che solo certi elementi vengano tenuti, numero di singular values automaticamente diventino meno.

controlla che tua simulazione MPS non sia PBC!


3. Convolution
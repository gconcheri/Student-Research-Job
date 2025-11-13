## TODO:

1. Show that chebyshev would work worse
We want to extract the noiseless version out of the noisy result -> 
out of Chebyshev: check for bigger interpolation power 
fare plot vs noisy function

fare plot vs noiseless function

2. convolution  

3. least square optimization with string tension


## Road map:

### Single site
TCI.ipynb --> TCI_singlesite, TCI_accumulative, TCI_evalvserror  --> Correlation_Function_TCI.ipynb

Chebyshev_interpolation_correct.ipynb --> Chebyshev.py: Chebyshev_interpolation, Chebyshev_interpolation_version2 -->  Correlation_function_Chebyshev.ipynb


### L sites
TCI_Lsite.ipynb --> TCI_Lsite, TCI_Lsite_accumulative.. --> Correlation_function_Lsites.ipynb, Correlation_function_models.ipynb, Correlation_function_TCI_MPS.ipynb, Corrfunc_LsiteaccumulativeTCI_relevant_plots_new.ipynb, Corrfunc_noisy_Lsite.ipynb


## other files
**ED_Cs_Lsites.py**: needed to create exact diagonalization simulation of Ising Model
**FourierTransform.py**: needed to do F.T. of correlation function to obtain spectral function (taken inspiration from MarkusFT)
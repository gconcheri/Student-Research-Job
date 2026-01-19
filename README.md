<!-- ## TODO:

1. Show that chebyshev would work worse
We want to extract the noiseless version out of the noisy result -> 
out of Chebyshev: check for bigger interpolation power 
fare plot vs noisy function

fare plot vs noiseless function

2. convolution  

3. least square optimization with string tension
-->

## Student Research Job

Hi everyone! Here you can find my code which I worked on to develop tensor networks algorithms potentially useful to **reduce computational costs** when having to measure correlation functions in **quantum computers** 🎸

<div class="alert alert-info">
<details>
<summary>For more details on the physical motivation (the abstract) click here</summary>
Measuring the correlation function of certain one-dimensional (1D) quantum systems is crucial for extracting the spectral function, which encodes key physical properties. Quantum computers offer significant potential for performing these measurements efficiently, as they can naturally simulate the time evolution of quantum systems. However, a major limiting factor is the **large number of distinct circuits** required to run on hardware to get a fine enough resolution of the correlation function to be able to extract the spectral function.

This project proposes an efficient method to reduce the number of distinct quantum circuit evaluations required to measure the correlation function at different times. This method leverages a generalization of interpolation algorithms, namely the Tensor Cross Interpolation (TCI) and Chebyshev algorithm. These algorithms enable the reconstruction of the correlation function–and consequently the spectral function–across different lattice sites and time steps with fewer function evaluations, and thus fewer measurements overall. To validate the approach, the methods are applied to the 1D transverse field Ising model in both the ferromagnetic and paramagnetic regimes, using classical simulations of noiseless and noisy quantum circuits. The results demonstrate their effectiveness, achieving a significant reduction in the number of measurements without compromising accuracy.
</details>
</div>


Simply put, these algorithms in general are useful to interpolate whichever 1D function using tensor networks. My work has been to generalize these algorithms to work also on 2D functions.  

---

# What's inside?
- **TCI :** contains all TCI related code.  
- **Chebyshev :** contains all Chebyshev related code.  
- **Correlation_function :** contains different jupyter notebooks where I apply the interpolation algorithms to check their efficiency and how well they work on the Transverse Field Ising Model.  
- **Pics:** some figures saved along the way from the different notebooks. 

---

Unfortunately, my code is a bit messy for now, so it might be a bit hard to go through it!

Here is a roadmap that may help guide the reading order needed to understand the different parts of the code 😉

## Road map:

### Single site (1D functions)
TCI.ipynb (Bernhard code) &rarr; TCI_singlesite...py, TCI_accumulative.py, TCI_evalvserror...py  &rarr; Correlation_Function_TCI.ipynb

Chebyshev_interpolation_correct.ipynb &rarr; Chebyshev.py: Chebyshev_interpolation, Chebyshev_interpolation_version2 &rarr;  Correlation_function_Chebyshev.ipynb


### L sites (generalization to 2D functions)
TCI_Lsite.ipynb &rarr; TCI_Lsite...py, TCI_Lsite_accumulative...py &rarr; Correlation_function_Lsites.ipynb, Correlation_function_models.ipynb, Correlation_function_TCI_MPS.ipynb, Corrfunc_Lsiteaccumulative...ipynb, TCI_relevant_plots_new.ipynb, Corrfunc_noisy_Lsite.ipynb

<div class="alert alert-info">
**TCI_Lsite_final.py** is the final generalized 2D TCI code.  
**Chebyshev.py** is the Chebyshev code, with both 1D and 2D Chebyshev algorithms. 
</div>



## Other files
**ED_Cs_Lsites.py**: needed to create exact diagonalization simulation of Ising Model
**FourierTransform.py**: needed to do F.T. of correlation function to obtain spectral function (taken inspiration from MarkusFT - code done by Markus Drescher) 

---

## Acknowledgements

Big thanks to Bernhard Jobst who supervised me throughout the whole project, and shared with me the 1d versions of both the TCI and Chebyshev algorithms to start from.

Thanks also to Markus Drescher for the Fourier Transform code

---

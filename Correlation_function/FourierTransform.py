import numpy as np
import scipy as sp


# Fourier transform the time-domain and space-domain with windowing function cos(pi/2 * t/T)**nw
# I built this function to manually do both fourier transforms, 
# check code if it does proper shifting by uncommenting

def FT_different(Ct, t_list, x_list, nw=4):
    """
    full complex data as input. 
    data only for positive time.
    """
    #Ct will now be matrix X x T
    
    n = len(t_list)
    Wfunlist = [np.cos(np.pi*t_list[t]/(2*t_list[-1]))**nw  for t in range(n)]
    a,b = Ct.shape 
    input_list = np.zeros((a,b), dtype = np.complex128)
    FTresult_t = np.zeros((a,b), dtype = np.complex128)
    FTresult = np.zeros((a,b), dtype = np.complex128)

    for i in range(a):
        input_list[i,:] = Wfunlist[:] * (np.array(Ct[i,:]))
        FTresult_t[i,:] = np.fft.fft(input_list[i,:])

    # freq_w = 2 * np.pi * np.fft.fftfreq(n, t_list[1]-t_list[0])
    # freq_w = np.fft.fftshift(freq_w)
    # for i in range(a):
    #     FTresult[i,:] = np.fft.fftshift(FTresult[i,:])

    for j in range(b):
        FTresult[:,j] = np.fft.fft(FTresult_t[:,j])
    
    # freq_k = 2 * np.pi * np.fft.fftfreq(n, x_list[1]-x_list[0])
    # freq_k = np.fft.fftshift(freq_k)

#     for j in range(b):
#         FTresult[:,j] = np.fft.fftshift(FTresult[:,j])
    
    return FTresult



# Fourier transform the time-domain and space-domain with windowing function cos(pi/2 * t/T)**nw using Scipy built-in function
#and no shifting!

import scipy.fft as fft

def FT(Ct, t_list, x_list, nw=4):
    """
    full complex data as input. 
    data only for positive time.
    """
    #Ct will now be matrix X x T
    
    n = len(t_list)
    Wfunlist = [np.cos(np.pi*t_list[t]/(2*t_list[-1]))**nw  for t in range(n)]
    a,b = Ct.shape 
    input_list = np.zeros((a,b), dtype = np.complex128)
    FTresult = np.zeros((a,b), dtype = np.complex128)

    for i in range(a):
        input_list[i,:] = Wfunlist[:] * (np.array(Ct[i,:]))
    
    FTresult = fft.fft2(input_list)
    #FTresult = fft.fftshift(FTresult)
    return FTresult




#Markus Drescher genius FT ahah

def fourier_time(t_series, dt, sigma = 0.4, nw=4, gauss = True):
    """ Calculates the FFT of a time series, applying a Gaussian window function. """

    # Gaussian or cosine window function
    n = len(t_series)
    t_list = np.arange(n)*dt

    if gauss == True:
        gauss = [np.exp(-1/2.*(i/(sigma * n))**2) for i in np.arange(n)]
        input_series = gauss * t_series
    else:
        Wfunlist = [np.cos(np.pi*t_list[t]/(2*t_list[-1]))**nw  for t in range(n)]
        input_series = Wfunlist * t_series

    # Fourier transform
    ft = np.fft.fft(input_series)
    freqs = np.fft.fftfreq(n, dt) * 2 * np.pi

    # order frequencies in increasing order
    end = np.argmin(freqs)
    freqs = np.append(freqs[end:], freqs[:end])

    # shift results accordingly
    ftShifted = np.append(ft[end:], ft[:end])

    # Take into account the additional minus sign in the time FT
    if len(ftShifted)%2 == 0:
        ftShifted = np.append(ftShifted, ftShifted[0])
        ftShifted = ftShifted[::-1]
        ftShifted = ftShifted[:-1]

    else:
        ftShifted = ftShifted[::-1]


    return freqs, ftShifted


def fourier_space(x_series):
    """ Calculates the FFT of a spatial series of values. """
    import numpy as np
    ft = np.fft.fft(x_series)
    n = len(x_series)
    momenta = 2*np.pi * np.fft.fftfreq(n, 1)

    # order momenta in increasing order
    momenta = np.fft.fftshift(momenta)

    # shift results accordingly
    Ck = np.fft.fftshift(ft)

    if n % 2 == 1:
        # extend the results to the whole Brillouin zone (right border included)
        momenta = np.append(momenta, -momenta[0])
        Ck = np.append(Ck, Ck[0])

    return momenta, Ck


def Swk(corrs, L, dt = 1e-2): #corrs as T x X matrix as input
    # Rearrange corrs such that position 0 corresponds to the perturbed site
    # (distance 0 to perturbation)

    xi = L//2
    c_temp = np.zeros(corrs.shape, dtype=complex)
    c_temp[:, :L-xi] = corrs[:, xi:]
    c_temp[:, L-xi:] = corrs[:, :xi]
    corrs = c_temp

    print('Compute Fourier transform')
    # Fourier transform in space
    if L % 2 == 0:
        corrs_tk = np.zeros((corrs.shape[0], corrs.shape[1]), dtype=complex)
    else:
        corrs_tk = np.zeros((corrs.shape[0], corrs.shape[1]+1), dtype=complex)
    for i in np.arange(corrs.shape[0]):
        momenta, Ck = fourier_space(corrs[i,:])
        corrs_tk[i, :] = Ck
    
    # Fourier transform in time
    Swk = np.zeros(corrs_tk.shape, dtype=complex)
    for k in np.arange(corrs_tk.shape[1]):
        freqs, Sw = fourier_time(corrs_tk[:, k], dt)
        Swk[:, k] = Sw
    print('finished')

    #print(freqs)

    return Swk
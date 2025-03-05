import numpy as np
import matplotlib.pyplot as plt



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

    if n % 2 == 0:
        # extend the results to the whole Brillouin zone (right border included)
        momenta = np.append(momenta, -momenta[0])
        Ck = np.append(Ck, Ck[0])

    return momenta, Ck


def get_Swk(corrs, L, dt = 1e-2): #corrs as T x X matrix as input
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
        corrs_tk = np.zeros((corrs.shape[0], corrs.shape[1]+1), dtype=complex)
    else:
        corrs_tk = np.zeros((corrs.shape[0], corrs.shape[1]), dtype=complex)
    for i in np.arange(corrs.shape[0]):
        momenta, Ck = fourier_space(corrs[i,:])
        corrs_tk[i, :] = Ck
    
    # Fourier transform in time
    Swk = np.zeros(corrs_tk.shape, dtype=complex)
    for k in np.arange(corrs_tk.shape[1]):
        freqs, Sw = fourier_time(corrs_tk[:, k], dt)
        Swk[:, k] = Sw
    print('finished')

    #Swk is of the form (W,K)

    return Swk, momenta, freqs

def plot_Swk(Swk, momenta, freqs, g = 2., J = 1., interval = 20, fig = (8,4), interp = False):
    plt.figure(figsize=fig)
    W, K = Swk.shape

    index = int(np.where(freqs == 0)[0])
    print(index)

    delta_K = (momenta[1]-momenta[0])/2
    K_min = momenta[0]-delta_K
    K_max = momenta[-1]+delta_K

    #Kmin = -Kmax
    # num. of momenta = K

    print(momenta.shape)
    print(K)

    delta_w = (freqs[index+1]-freqs[index])/2
    W_min = freqs[index]-delta_w
    W_max = freqs[index+interval]+delta_w

    # sel_freqs = [W_min, W_max]
    # sel_momenta = [K_min, K_max]
    # sel_momenta_idx = [0,K-1]
    # sel_freqs_idx = [0, interval-1]

    # plt.xticks(sel_momenta_idx, sel_momenta)
    # plt.yticks(sel_freqs_idx, sel_freqs)

    plt.imshow(np.abs(Swk[index:(index+interval+1), :]), aspect = 'auto', 
            interpolation = 'none',
            origin='lower', 
            cmap='inferno',
            extent = [K_min, K_max, W_min, W_max]
            )


    omega = 2*g - 2 * J * np.cos(momenta)  # The dispersion relation
    plt.plot(momenta, omega, color='yellow', linestyle='dotted', linewidth=1.5, label='dispersion relation E(k)')

    plt.colorbar(fraction=0.046, pad=0.04)
    if interp == True:
        plt.title(r'abs ED $Cs(\omega,k)$')
    else:
        plt.title(r'abs ID $Cs(\omega,k)$')

    plt.xlim(momenta[0], momenta[-1])

    plt.xlabel('momentum k')
    plt.ylabel(r'frequency $\omega$')

    plt.xticks(momenta)

    plt.legend()
    plt.plot()

#here idx_model represents the index (either 0,1,2,3) referred to one the four models 
def fig_Swk(Swk, momenta, freqs, interp_Swk, interp_momenta, interp_freqs, idx_model, g_par,
                g = 2., J = 1., interval = 20):
    
    #W, K = Swk.shape

    index = int(np.where(freqs == 0)[0])
    interp_index = int(np.where(interp_freqs == 0)[0])

    delta_K = (momenta[1]-momenta[0])/2
    K_min = momenta[0]-delta_K
    K_max = momenta[-1]+delta_K

    delta_w = (freqs[index+1]-freqs[index])/2
    W_min = freqs[index]-delta_w
    W_max = freqs[index+interval]+delta_w

    delta_Kinterp = (interp_momenta[1]-interp_momenta[0])/2
    Kinterp_min = interp_momenta[0]-delta_Kinterp
    Kinterp_max = interp_momenta[-1]+delta_Kinterp

    delta_winterp = (interp_freqs[index+1]-interp_freqs[index])/2
    Winterp_min = interp_freqs[interp_index] - delta_winterp
    Winterp_max = interp_freqs[interp_index+interval] + delta_winterp

    Swk = np.abs(Swk[index:(index+interval+1), :])
    interp_Swk = np.abs(interp_Swk[index:(index+interval+1), :])

    omega = disp_relation(g_par = g_par, momenta = momenta, idx_model = idx_model)

    # Data and titles for each subplot
    data = [
        (Swk, r'abs ED $Cs(\omega,k)$', [K_min, K_max, W_min, W_max]),
        (interp_Swk, r'abs ID $Cs(\omega,k)$', [Kinterp_min, Kinterp_max, Winterp_min, Winterp_max]),
        (np.abs(Swk-interp_Swk), 'abs theo - iter', [K_min, K_max, W_min, W_max])
    ]

    rows, cols = 1, 3  # Define grid dimensions

    # Create subplots
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))

    ee = 0 
    # Loop through data and subplots
    for ax, (image, title, extent) in zip(axs.flat, data):
        ee += 1
        im = ax.imshow(image, aspect='auto', 
                        interpolation='none', 
                        origin = 'lower', 
                        cmap = 'inferno',
                        extent = extent)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Add colorbar
        ax.set_title(title)
        ax.set(xlabel = 'momentum (k)', ylabel = r'frequency ($\omega$)')
        ax.set(xlim = [momenta[0], momenta[-1]])

        if ee < 3:
            if idx_model<2: 
                ax.plot(momenta, omega, color='orange', linestyle='--', linewidth=2, label='dispersion relation E(k)')
            else:
                ax.plot(momenta, omega[0], color='orange', linestyle='--', linewidth=2, label='lower boundary E(k)')
                ax.plot(momenta, omega[1], color='orange', linestyle='dotted', linewidth=2, label='upper boundary E(k)')
        ax.legend()

    plt.tight_layout()
    plt.show()


def disp_relation(momenta, idx_model = None, g =2., J =1., k = 0.1, g_par = 0.5): # Calculates the theoretically expected dispersion relation for each model (esatta per model= 0,1 , per model 2,3 non proprio)

    if idx_model==0 or idx_model == None:
        return 2 * g - 2 * J * np.cos(momenta)  
    elif idx_model==1:
        return 4*k + 2 * g - 2 * J * np.cos(momenta)  
    else:
        return [4* J - 4 * g_par * np.cos(momenta/2), 4* J + 4* g_par * np.cos(momenta/2)]



#     #Jupyter cell to plot Skw by using functions FT or FT_different
# #It's a first not refined way of plotting without centering the plot, without placing the extents and so on

# D = L
# a, b = Cs.shape
# interp_Cs = func_interp.T.reshape(a, b)

# FTresult = FT(Ct=Cs, t_list=np.arange(N)*dt, x_list = np.arange(D), nw=3)
# FTresult_i = FT(Ct=interp_Cs, t_list=np.arange(N)*dt, x_list = np.arange(D), nw=3)

# print(FTresult.shape) #FT result is of the form (K, W)
# print(FTresult_i.shape)

# FTresult = FTresult.T #now FT result is of the form (W, K)
# FTresult_i = FTresult_i.T

# # Define the threshold
# threshold = 30

# # Discard rows where all elements are below the threshold
# rows_mask = np.any(np.abs(FTresult) >= threshold, axis=1)  # Check if any element in a row meets the threshold
# FTresult = FTresult[rows_mask, :]
# FTresult_i = FTresult_i[rows_mask, :]

# # Discard columns where all elements are below the threshold
# columns_mask = np.any(np.abs(FTresult) >= threshold, axis=0)  # Check if any element in a column meets the threshold
# FTresult = FTresult[:, columns_mask]
# FTresult_i = FTresult_i[:, columns_mask]


# rows, cols = 2, 3  # Define grid dimensions


# # Data and titles for each subplot
# data = [
#     (np.real(FTresult), r'real ED $Cs(\omega,k)$'),
#     (np.real(FTresult_i), r'real ID $Cs(\omega,k)$'),
#     (np.real(FTresult) - np.real(FTresult_i), 'real theo - inter'),
#     (np.imag(FTresult), r'imaginary ED $Cs(\omega,k)'),
#     (np.imag(FTresult_i), r'imaginary ID $Cs(\omega,k)$'),
#     (np.imag(FTresult) - np.imag(FTresult_i), 'imag theo - inter')
# ]

# # Create subplots
# fig, axs = plt.subplots(rows, cols, figsize=(16, 8))

# # Loop through data and subplots
# for ax, (image, title) in zip(axs.flat, data):
#     im = ax.imshow(image, aspect='auto', interpolation='none')
#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Add colorbar
#     ax.set_title(title)

# plt.tight_layout()
# plt.show()

#---------------------------------------------------


#old way of doing Fourier Transform, added to TCI and Chebyshev loops 
# in jupyter file 'Correlation_function_models.ipynb'

# FTresult = FT.FT(Ct=Cs, t_list=np.arange(N)*dt, x_list = np.arange(D), nw=3)
# FTresult_i = FT.FT(Ct=interp_Cs, t_list=np.arange(N)*dt, x_list = np.arange(D), nw=3)

# FTresult = FTresult.T
# FTresult_i = FTresult_i.T

# # Discard rows where all elements are below the threshold
# rows_mask = np.any(np.abs(FTresult) >= threshold, axis=1)  # Check if any element in a row meets the threshold
# FTresult = FTresult[rows_mask, :]
# FTresult_i = FTresult_i[rows_mask, :]

# # # Discard columns where all elements are below the threshold
# columns_mask = np.any(np.abs(FTresult) >= threshold, axis=0)  # Check if any element in a column meets the threshold
# FTresult = FTresult[:, columns_mask]
# FTresult_i = FTresult_i[:, columns_mask]

# # Data and titles for each subplot
# data = [
#     #(np.real(FTresult), r'real ED $Cs(\omega,k)$'),
#     #(np.real(FTresult_i), r'real ID $Cs(\omega,k)$'),
#     #(np.real(FTresult) - np.real(FTresult_i), 'real theo - inter'),
#     #(np.imag(FTresult), r'imaginary ED $Cs(\omega,k)$'),
#     #(np.imag(FTresult_i), r'imaginary ID $Cs(\omega,k)$'),
#     #(np.imag(FTresult) - np.imag(FTresult_i), 'imag theo - inter'),
#     (np.abs(FTresult), r'abs ED $Cs(\omega,k)$'),
#     (np.abs(FTresult_i), r'abs ID $Cs(\omega,k)$'),
#     (np.abs(FTresult)-np.abs(FTresult_i), 'abs theo - iter')
# ]

# rows, cols = 1, 3  # Define grid dimensions

# # Create subplots
# fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))

# # Loop through data and subplots
# for ax, (image, title) in zip(axs.flat, data):
#     im = ax.imshow(image, aspect='auto', interpolation='none')
#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # Add colorbar
#     ax.set_title(title)
# plt.tight_layout()
# plt.show()
import b_model_Gio
import c_tebd_Gio
import numpy as np
import pickle
import os
import warnings

sigmay = np.array([[0, -1j], [1j, 0]])
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]])


def _validate_inputs(L, n, dt, site, X, Y):
    if L < 2:
        raise ValueError("L must be >= 2.")
    if n < 0:
        raise ValueError("n must be >= 0.")
    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if site is not None and not (0 <= site < L):
        raise ValueError(f"site must satisfy 0 <= site < L (got site={site}, L={L}).")
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.shape != (2, 2) or Y.shape != (2, 2):
        raise ValueError(f"X and Y must be 2x2 operators (got X{X.shape}, Y{Y.shape}).")


def construct_intermediate_tensors(psi, psi1, L):
    tensor1 = [np.array([[1]])]
    tensor1.append(np.tensordot(psi.Bs[L - 1], psi1.Bs[L - 1].conj(), [[1, 2], [1, 2]])) # vL [i] [vR] , vL* [i*] [vR*] -> vL vL*
    for j in range(L - 2, 0, -1):
        tensor2 = np.tensordot(psi.Bs[j], tensor1[-1], [2, 0]) # vL i [vR] ,[vL] vL* -> vL i vL*
        tensor1.append(np.tensordot(tensor2, psi1.Bs[j].conj(), [[1, 2], [1, 2]])) # vL [i] [vL*], vL* [i*] [vR*] ->  vL vL*
    return tensor1


def correlation_allsites(psi, psi1, X, L):

    tensor1 = construct_intermediate_tensors(psi,psi1,L)

    corr_list = []

    tensorA = np.tensordot(psi.Bs[0], psi1.Bs[0].conj(),[[0,1],[0,1]]) # [vL] [i] vR , [vL*] [i*] vR* -> vR vR*
    tensorB = np.tensordot(psi.Bs[0], X, [1,1]) # vL [i] vR , i [i*] -> vL vR i
    tensorC = np.tensordot(tensorB,psi1.Bs[0].conj(), [[0,2],[0,1]]) # [vL] vR [i], [vL*] [i*] vR* -> vR vR*
    corr_list.append(np.tensordot(tensorC,tensor1[L-1], [[0,1],[0,1]])) # [vR] [vR*], [vL] [vL*]

    for j in range(1,L):
        tensorB = np.tensordot(psi.Bs[j], X, [1,1]) # vL [i] vR , i [i*] -> vL vR i
        tensorbeforeC = np.tensordot(tensorA, tensorB, [0,0]) # [vR] vR*, [vL] vR i -> vR* vR i
        tensorC = np.tensordot(tensorbeforeC,psi1.Bs[j].conj(), [[0,2],[0,1]]) # [vR*] vR [i], [vL*] [i*] vR* -> vR vR*
        #print(tensorC.shape)
        #print(tensor1[L-j-2].shape) #- mettili se c'è sum mismatch prima di un certo tensorproduct
        corr_list.append(np.tensordot(tensorC,tensor1[L-j-1], [[0,1],[0,1]])) # [vR] [vR*], [vL] [vL*]
        #tensorintermediate = np.tensordot(psi1.Bs[j], psi.Bs[j].conj(),[1,1]) # vL [i] vR , vL* [i*] vR* -> vL vR vL* vR*
        #tensorloop = np.tensordot(tensorA, tensorintermediate, [[0,1],[0,2]]) # [vR] [vR*], [vL] vR [vL*] vR* -> vR vR*
        tensorintermediate = np.tensordot(tensorA, psi.Bs[j], [0,0]) # [vR] vR*, [vL] i vR -> vR* i vR
        tensorloop = np.tensordot(tensorintermediate, psi1.Bs[j].conj(), [[0,1],[0,1]]) # [vR*] [i] vR, [vL*] [i*] vR* -> vR vR*
        tensorA = tensorloop    

    return corr_list


def correlation_Ctj(L, J, g, X, Y, n, dt, k = 0.1, h = 0, chi_max = 200, site = None,
                    savedir = "simulation", ops_name = "sigmay"):
    
    _validate_inputs(L, n, dt, site, X, Y)

    if g>J: #paramagnetic case
        E0, psi, model = c_tebd_Gio.example_TEBD_gs_finite(L,J,g,h,k)
    else: #ferromagnetic case -> have to find unique gs
        E0, psi, _ = c_tebd_Gio.example_TEBD_gs_finite(L,J,g,h=10**-2.5,k=k)
        model = b_model_Gio.TFIModel(L,J=J,g=g,h=h,k=k)

        #check magnetization
        sigmax = np.array([[0,1],[1,0]])
        x_value = correlation_allsites(psi, psi, sigmax, L)
        mean_x = np.real_if_close(np.mean(x_value))
        print(mean_x)
        if mean_x > 0:
            print("Ground state is the one with positive magnetization")
        else:
            print("Ground state is the one with negative magnetization")

    psi1 = psi.copy() #take a copy of psi which is ground state
    S = []
    if site is None:
        site = L//2 # if no site is specified, take the middle one
    tensor1 = np.tensordot(psi.Bs[site], Y, [1, 1]) # vL [i] vR, i [i*] -> vL vR i
    psi.Bs[site] = tensor1.transpose([0, 2, 1])
    
    U_bond = c_tebd_Gio.calc_U_bonds_real(model, dt)
    result = []
    N=2**n

    for r in range(N):
        exp_factor = np.exp(1j*E0*r*dt)
        
        if r != 0:
            c_tebd_Gio.run_TEBD(psi, U_bond, N_steps=1, chi_max=chi_max, eps=1.e-10) #found new psi by applying e^-iHdt
        S.append(psi.entanglement_entropy())

        corr_list = correlation_allsites(psi, psi1, X, L)
        result.append(np.asarray(corr_list) * exp_factor)

    phase = "para" if g > J else "ferro"
    subfold = (
        f"L{L}"
        + f"_{ops_name}"
        + f"_n{n}"
        + f"_dt{dt}"
        + f"_g{g}"
        + f"_J{J}"
        + f"_k{k}"
        + f"_h{h}"
        + f"_chi{chi_max}"
    )

    outdir = os.path.join(savedir, phase, subfold)
    fpath_s = os.path.join(outdir, "S.pkl")
    fpath_corr = os.path.join(outdir, "Corr.pkl")

    if os.path.exists(fpath_s) or os.path.exists(fpath_corr):
        warnings.warn(
            f"Output path already exists. Files may be overwritten:\n{fpath_s}\n{fpath_corr}",
            stacklevel=2,
        )

    os.makedirs(outdir, exist_ok=True)

    with open(fpath_s, "wb") as f:
        pickle.dump(S, f)

    with open(fpath_corr, "wb") as f:
        pickle.dump(result, f)


def simulation(**kwargs):
    L = kwargs.get('L', 55)
    J = kwargs.get('J', 1.)
    g = kwargs.get('g', 0.15)
    X = kwargs.get('X', sigmay)
    Y = kwargs.get('Y', sigmay)
    n = kwargs.get('n', 12)
    dt = kwargs.get('dt', 0.01)
    k = kwargs.get('k', 0.1)
    h = kwargs.get('h', 0)
    chi_max = kwargs.get('chi_max', 200)
    site = kwargs.get('site', None)
    savedir = kwargs.get('savedir', "simulations")
    ops_name = kwargs.get('ops_name', "sigmay")

    correlation_Ctj(
        L, J, g, X, Y, n, dt, k=k, h=h, chi_max=chi_max, site=site,
        savedir=savedir, ops_name=ops_name
    )
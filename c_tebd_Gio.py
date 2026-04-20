"""Toy code implementing the time evolving block decimation (TEBD)."""

import numpy as np
from scipy.linalg import expm
from a_mps_Gio import split_truncate_theta


def calc_U_bonds(model, dt):
    """Given a model, calculate ``U_bonds[i] = expm(-dt*model.H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means imaginary time evolution!
    """
    H_bonds = model.H_bonds
    d = H_bonds[0].shape[0]
    U_bonds = []
    for H in H_bonds:
        H = np.reshape(H, [d * d, d * d])
        U = expm(-dt * H)
        U_bonds.append(np.reshape(U, [d, d, d, d]))
    return U_bonds


def calc_U_bonds_real(model, dt):
    """Given a model, calculate ``U_bonds[i] = expm(-idt*model.H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    """
    H_bonds = model.H_bonds
    d = H_bonds[0].shape[0]
    U_bonds = []
    for H in H_bonds:
        H = np.reshape(H, [d * d, d * d])
        U = expm(- 1j * dt * H)
        U_bonds.append(np.reshape(U, [d, d, d, d]))
    return U_bonds


def run_TEBD(psi, U_bonds, N_steps, chi_max, eps):
    """Evolve the state `psi` for `N_steps` time steps with (first order) TEBD.

    The state psi is modified in place."""
    Nbonds = psi.L - 1
    assert len(U_bonds) == Nbonds
    for n in range(N_steps):
        for k in [0, 1]:  # even, odd
            for i_bond in range(k, Nbonds, 2):
                update_bond(psi, i_bond, U_bonds[i_bond], chi_max, eps)
    # done


def update_bond(psi, i, U_bond, chi_max, eps):
    """Apply `U_bond` acting on i,j=(i+1) to `psi`."""
    j = i + 1
    # construct theta matrix
    theta = psi.get_theta2(i)  # vL i j vR
    # apply U
    Utheta = np.tensordot(U_bond, theta, axes=([2, 3], [1, 2]))  # i j [i*] [j*], vL [i] [j] vR
    Utheta = np.transpose(Utheta, [2, 0, 1, 3])  # vL i j vR
    # split and truncate
    Ai, Sj, Bj = split_truncate_theta(Utheta, chi_max, eps)
    # put back into MPS
    Gi = np.tensordot(np.diag(psi.Ss[i]**(-1)), Ai, axes=[1, 0])  # vL [vL*], [vL] i vC
    psi.Bs[i] = np.tensordot(Gi, np.diag(Sj), axes=[2, 0])  # vL i [vC], [vC] vC
    psi.Ss[j] = Sj  # vC
    psi.Bs[j] = Bj  # vC j vR



def example_TEBD_gs_finite(L, J, g, h = 0, k = 0):
    print("finite TEBD, (imaginary time evolution)")
    print("L={L:d}, J={J:.1f}, g={g:.2f}, h = {h}, k = {k:.3f}".format(L=L, J=J, g=g, h=h,k=k))
    import a_mps_Gio
    import b_model_Gio
    model = b_model_Gio.TFIModel(L, J=J, g=g, h=h, k=k)
    psi = a_mps_Gio.init_spinup_MPS(L)
    for dt in [0.1, 0.01, 0.001, 1.e-4, 1.e-5]:
        U_bonds = calc_U_bonds(model, dt)
        run_TEBD(psi, U_bonds, N_steps=500, chi_max=100, eps=1.e-10)
        E = model.energy(psi)
        print("dt = {dt:.5f}: E = {E:.13f}".format(dt=dt, E=E))
    print("final bond dimensions: ", psi.get_chi())
    return E, psi, model


if __name__ == "__main__":
    example_TEBD_gs_finite(L=14, J=1., g=1.5)

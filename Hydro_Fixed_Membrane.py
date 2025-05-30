import numpy as np
from scipy.sparse import coo_matrix

from scipy.sparse import eye, kron
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator

import matplotlib.tri as mtri

import time
import sys

import numba as nmba
from numba import njit, jit, prange

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from libMobility import NBody


def main():
    # load the N x3 faces matrix in a file called 'faces.txt'
    T = np.loadtxt("faces.txt", dtype=int)
    # subtract 1 from T to convert from 1-based to 0-based indexing
    T -= 1
    # load the N x3 vertex coordinates in a file called 'free_vertex.txt'
    V_free = np.loadtxt("free_vertex.txt", dtype=float)
    # load the N x3 fixed vertex coordinates in a file called 'fixed_vertex.txt'
    V_fixed = np.loadtxt("fixed_vertex.txt", dtype=float)
    # concatenate the free and fixed vertices
    V = np.vstack((V_free, V_fixed))
    # number of free vertices
    Nfree = V_free.shape[0]
    N = V.shape[0]

    # zero out the z-component of V
    V[:, 2] = 0.0
    print(V)

    # get the diamond neighbor list
    tri_nbs = get_diamonds(T, V)

    # plot
    fig, ax = plot_mesh(V, V_fixed, T)

    # find mesh size as the min distance between two vertices
    mesh_size = np.inf
    for i in range(V.shape[0]):
        for j in range(i + 1, V.shape[0]):
            dist = np.linalg.norm(V[i] - V[j])
            if dist < mesh_size:
                mesh_size = dist

    kbt = 0.0
    eta = 1.0
    hydroRadius = 0.5 * mesh_size

    solver = NBody("open", "open", "open")
    solver.setParameters()
    solver.initialize(temperature=kbt, viscosity=eta, hydrodynamicRadius=hydroRadius)

    dt = 30 * (mesh_size**3)  # time step size

    Final_time = 20.0  # final time
    Nsteps = int(Final_time / dt)  # number of time steps

    for step in range(Nsteps):
        # --- Implicit update: solve for dX --- # TODO outdated, update the equation
        #    ( I + dt*Kx ) dX = -dt * ( Kx * X_free + RHSf - F_push )

        #### build membrane forces
        # bending force
        row, col, val, bending_boundary_contribution = assemble_willmore(tri_nbs, V, Nfree)
        D = coo_matrix((val, (row, col)), shape=(Nfree, Nfree)).tocsr()
        Dx = kron(D, eye(3, format="csr")) # D is (Nfree×Nfree); Dx will be (3*Nfree×3*Nfree)

        # bending force
        X0 = V.flatten()
        F_bend = Dx.dot(X0[: 3 * Nfree])

        # inhomogeneous boundary contribution from bending
        bending_bc_free = bending_boundary_contribution[: 3 * Nfree].copy()  # Truncate RHS to the free‐DOFs only:

        # pushing force on middle of membrane
        F_push = np.zeros(3 * Nfree, dtype=float)
        F_push[2] = -1.0 * max(0.0, 10.0 - dt * step)

        membrane_forces = - (F_bend + bending_bc_free + F_push)

        start = time.time()
        RHS = np.zeros(3 * (N + Nfree), dtype=float)
        RHS[3*N : ] = membrane_forces

        Bend_mat = dt * Dx

        solver.setPositions(V)
        def apply_A(vec):
            lam = vec[0 : 3 * N]
            l_free = lam[0 : 3 * Nfree]

            u = np.zeros_like(lam)
            u[0 : 3 * Nfree] = vec[3 * N :]

            Mf, _ = solver.Mdot(forces=lam)

            out = np.zeros(3 * (N + Nfree))
            out[0 : 3 * N] = Mf - u
            out[3 * N :] = l_free + Bend_mat.dot(u[0 : 3 * Nfree])

            return out

        A = LinearOperator((3 * (N + Nfree), 3 * (N + Nfree)), matvec=apply_A)

        sol, info = gmres(A, RHS, rtol=1e-4, x0=None, restart=100, maxiter=1000)
        Ufree = sol[: 3 * Nfree]  # only the free DOFs
        end = time.time()
        print(f"Solving took {end - start:.4f} seconds")

        # Compute new X
        X1 = X0.copy()
        X1[: 3 * Nfree] = X0[: 3 * Nfree] + dt*Ufree

        V = X1.reshape((-1, 3))
        if step % 100 == 0:
            print(f"Step {step}, plotting")
            fig, ax = plot_mesh(V, V[Nfree::, :], T, fig, ax, i=step)


def get_diamonds(T, V):
    """
    Build the “diamond” neighbor list for each internal edge of a triangulation.

    Parameters
    ----------
    T : (nt, 3) int array
        Zero-based triangle‐connectivity, one row per triangle.
    V : (nv, d) array
        Vertex coordinates (only used here to infer nv = number of vertices).
    N_fix : int
        Number of fixed vertices at the end of the V‐array (not used here).
    N_adj : int
        Number of adjacent‐to‐boundary vertices (not used here).

    Returns
    -------
    tri_nbs : (nv*nv, 4) float array
        For each edge‐key, up to four vertex‐indices [v1, v2, v3, v4] forming
        a “diamond” across that edge.  Entries remain NaN if the second
        triangle hasn’t been seen yet.
    """
    nt = T.shape[0]
    nv = V.shape[0]

    # same size as nv^2, track first/second occurrence
    col_ind = np.ones(nv * nv, dtype=int)

    # initialize with NaNs
    tri_nbs = np.full((nv * nv, 4), np.nan, dtype=float)

    # loop over triangles
    for tri in T:
        i, j, k = tri  # zero-based

        # for each of the three edges, sort its endpoints
        e1 = np.sort([i, j])
        e2 = np.sort([j, k])
        e3 = np.sort([k, i])

        # compute a unique key in [0, nv*nv):
        #   key = small_index * nv + large_index
        key1 = e1[0] * nv + e1[1]
        key2 = e2[0] * nv + e2[1]
        key3 = e3[0] * nv + e3[1]

        # if first time seeing this edge, store [j,k,i] etc.
        if col_ind[key1] == 1:
            tri_nbs[key1, 0:3] = [j, k, i]
            col_ind[key1] = 2
        else:
            tri_nbs[key1, 3] = k

        if col_ind[key2] == 1:
            tri_nbs[key2, 0:3] = [k, i, j]
            col_ind[key2] = 2
        else:
            tri_nbs[key2, 3] = i

        if col_ind[key3] == 1:
            tri_nbs[key3, 0:3] = [i, j, k]
            col_ind[key3] = 2
        else:
            tri_nbs[key3, 3] = j

    # remove rows that never got both triangles (optionally)
    mask = ~np.isnan(tri_nbs).any(axis=1)
    tri_nbs = tri_nbs[mask]
    tri_nbs = tri_nbs.astype(int)  # convert to int

    return tri_nbs


def set_grad_beta(V, row, col, val, k, j, l, i, Nfree, RHS, tol=1e-5):
    """
    Port of the MATLAB set_grad_beta to Python/NumPy.

    Arguments
    ---------
    V    : (nv,3) float array of vertex coords
    row  : list, to which we append row‐indices
    col  : list, to which we append column‐indices
    val  : list, to which we append values
    k,j,l,i : ints, zero‐based vertex indices in the diamond
    Nfree   : int, number of free vertices (others are fixed)
    RHS     : (3*nv,) float array, right‐hand side vector to increment
    tol     : small threshold for sin(beta)

    Returns
    -------
    Nothing: row/col/val lists and RHS array are modified in place.
    """

    # Grab the four positions
    vk = V[k, :]
    vj = V[j, :]
    vl = V[l, :]
    vi = V[i, :]

    # Edge vectors and unit directions
    a = vj - vk
    ma = np.linalg.norm(a)
    A = a / ma

    b = vl - vj
    mb = np.linalg.norm(b)
    B = b / mb

    c = vi - vl
    mc = np.linalg.norm(c)
    C = c / mc

    d = vk - vi
    md = np.linalg.norm(d)
    D = d / md

    # Compute cos(beta) via your formula
    cosb = (
        np.dot(A, C) * np.dot(B, D)
        - np.dot(A, B) * np.dot(C, D)
        - np.dot(B, C) * np.dot(D, A)
    )
    beta = np.arccos(cosb)
    sinb = np.sin(beta)

    if abs(sinb) <= tol:
        return

    # The four coefficients
    c_a = (-1.0 / sinb) * (cosb / (ma**2) - np.dot(B, C) / (ma * md))
    c_b = (-1.0 / sinb) * (np.dot(A, C) / (mb * md) + np.dot(C, D) / (mb * ma))
    c_c = (1.0 / sinb) * (np.dot(A, B) / (mc * md) + np.dot(B, D) / (mc * ma))
    c_d = (1.0 / sinb) * (cosb / (md**2) - np.dot(B, C) / (ma * md))

    # The slice of RHS corresponding to vertex k's 3 DOFs
    I_rhs = slice(3 * k, 3 * k + 3)

    # If k is a free vertex, we add entries to the sparse matrix
    if k < Nfree:
        # diagonal term (k,k)
        row.append(k)
        col.append(k)
        val.append(c_d - c_a)

        # off‐diagonals (with j, l, i) or move to RHS if fixed:
        #   (k,j)
        if j < Nfree:
            row.append(k)
            col.append(j)
            val.append(c_a - c_b)
        else:
            RHS[I_rhs] += vj * (c_a - c_b)

        #   (k,l)
        if l < Nfree:
            row.append(k)
            col.append(l)
            val.append(c_b - c_c)
        else:
            RHS[I_rhs] += vl * (c_b - c_c)

        #   (k,i)
        if i < Nfree:
            row.append(k)
            col.append(i)
            val.append(c_c - c_d)
        else:
            RHS[I_rhs] += vi * (c_c - c_d)

    # else:
    # if k itself is fixed, move the entire diagonal contribution into RHS
    #### TODO: This is likely pointless - Someone should try removing it
    #    RHS[I_rhs] += vk * (c_d - c_a)


def willmore_mat_alt(tri_nbs, V, Nfree):
    """
    Build the Willmore stiffness matrix K and RHS vector for the
    “alternative” formulation.

    Parameters
    ----------
    tri_nbs : (nd, 4) array
        Each row is [j, k, i, l] (zero‐based) describing one “diamond.”
    V       : (nv, 3) float array
        Vertex coordinates.
    Nfree   : int
        Number of free vertices; global system is size (3*Nfree).

    Returns
    -------
    K   : (3*Nfree, 3*Nfree) csr_matrix
        The assembled sparse stiffness matrix.
    RHS : (3*nv,) float array
        Right‐hand side vector (fixed‐DOF contributions included).
    """
    nd, _ = tri_nbs.shape
    nv, _ = V.shape

    # Lists to accumulate sparse entries
    row = []
    col = []
    val = []

    # Initialize RHS for all vertices (we only use first 3*Nfree entries ultimately)
    RHS = np.zeros(3 * nv, dtype=float)

    # Loop over each “diamond”
    for dia in range(nd):
        idx = tri_nbs[dia]  # [j, k, i, l] zero‐based

        # reassign to match your MATLAB order
        k, j, l, i = idx[1], idx[0], idx[3], idx[2]
        # print(f"Processing diamond: {k}, {j}, {l}, {i}")

        # call set_grad_beta on each cyclic rotation
        set_grad_beta(V, row, col, val, k, j, l, i, Nfree, RHS)
        set_grad_beta(V, row, col, val, j, l, i, k, Nfree, RHS)
        set_grad_beta(V, row, col, val, l, i, k, j, Nfree, RHS)
        set_grad_beta(V, row, col, val, i, k, j, l, Nfree, RHS)

    # Build sparse matrix (only the free‐DOF block of size Nfree × Nfree)
    K = coo_matrix((val, (row, col)), shape=(Nfree, Nfree)).tocsr()

    return K, RHS


########################################
@njit(fastmath=True)
def assemble_willmore(tri_nbs, V, Nfree, tol=1e-5):
    """
    Faster single-pass Willmore assembly by pre-allocating arrays.
    """
    nv = V.shape[0]
    nd = tri_nbs.shape[0]

    # upper bound: 16 entries per diamond
    max_entries = nd * 16
    row = np.empty(max_entries, dtype=np.int64)
    col = np.empty(max_entries, dtype=np.int64)
    val = np.empty(max_entries, dtype=np.float64)
    ptr = 0

    RHS = np.zeros(3 * nv, dtype=np.float64)

    # inline norm for speed
    def norm3(u0, u1, u2):
        return np.sqrt(u0 * u0 + u1 * u1 + u2 * u2)

    for d in range(nd):
        j, k, i, l = tri_nbs[d, 0], tri_nbs[d, 1], tri_nbs[d, 2], tri_nbs[d, 3]
        for ka, ja, la, ia in ((k, j, l, i), (j, l, i, k), (l, i, k, j), (i, k, j, l)):
            vk = V[ka]
            vj = V[ja]
            vl = V[la]
            vi = V[ia]
            # compute normalized edges
            a0, a1, a2 = vj[0] - vk[0], vj[1] - vk[1], vj[2] - vk[2]
            ma = norm3(a0, a1, a2)
            A0, A1, A2 = a0 / ma, a1 / ma, a2 / ma
            b0, b1, b2 = vl[0] - vj[0], vl[1] - vj[1], vl[2] - vj[2]
            mb = norm3(b0, b1, b2)
            B0, B1, B2 = b0 / mb, b1 / mb, b2 / mb
            c0, c1, c2 = vi[0] - vl[0], vi[1] - vl[1], vi[2] - vl[2]
            mc = norm3(c0, c1, c2)
            C0, C1, C2 = c0 / mc, c1 / mc, c2 / mc
            d0, d1, d2 = vk[0] - vi[0], vk[1] - vi[1], vk[2] - vi[2]
            md = norm3(d0, d1, d2)
            D0, D1, D2 = d0 / md, d1 / md, d2 / md

            # dot products
            dotAC = A0 * C0 + A1 * C1 + A2 * C2
            dotBD = B0 * D0 + B1 * D1 + B2 * D2
            dotAB = A0 * B0 + A1 * B1 + A2 * B2
            dotCD = C0 * D0 + C1 * D1 + C2 * D2
            dotBC = B0 * C0 + B1 * C1 + B2 * C2
            dotDA = D0 * A0 + D1 * A1 + D2 * A2

            cosb = dotAC * dotBD - dotAB * dotCD - dotBC * dotDA
            # clamp
            if cosb > 1.0:
                cosb = 1.0
            if cosb < -1.0:
                cosb = -1.0
            beta = np.arccos(cosb)
            sinb = np.sin(beta)
            if abs(sinb) <= tol:
                continue

            ca = (-1.0 / sinb) * (cosb / (ma * ma) - dotBC / (ma * md))
            cb = (-1.0 / sinb) * (dotAC / (mb * md) + dotCD / (mb * ma))
            cc = (1.0 / sinb) * (dotAB / (mc * md) + dotBD / (mc * ma))
            cd = (1.0 / sinb) * (cosb / (md * md) - dotBC / (ma * md))

            base = 3 * ka
            if ka < Nfree:
                # diag
                row[ptr] = ka
                col[ptr] = ka
                val[ptr] = cd - ca
                ptr += 1
                # (ka,ja)
                if ja < Nfree:
                    row[ptr] = ka
                    col[ptr] = ja
                    val[ptr] = ca - cb
                    ptr += 1
                else:
                    RHS[base] += vj[0] * (ca - cb)
                    RHS[base + 1] += vj[1] * (ca - cb)
                    RHS[base + 2] += vj[2] * (ca - cb)
                # (ka,la)
                if la < Nfree:
                    row[ptr] = ka
                    col[ptr] = la
                    val[ptr] = cb - cc
                    ptr += 1
                else:
                    RHS[base] += vl[0] * (cb - cc)
                    RHS[base + 1] += vl[1] * (cb - cc)
                    RHS[base + 2] += vl[2] * (cb - cc)
                # (ka,ia)
                if ia < Nfree:
                    row[ptr] = ka
                    col[ptr] = ia
                    val[ptr] = cc - cd
                    ptr += 1
                else:
                    RHS[base] += vi[0] * (cc - cd)
                    RHS[base + 1] += vi[1] * (cc - cd)
                    RHS[base + 2] += vi[2] * (cc - cd)
            else:
                RHS[base] += vk[0] * (cd - ca)
                RHS[base + 1] += vk[1] * (cd - ca)
                RHS[base + 2] += vk[2] * (cd - ca)

    # trim arrays to actual size
    return row[:ptr], col[:ptr], val[:ptr], RHS


########################################


def plot_mesh(
    V,
    V_fixed,
    T,
    fig=None,
    ax=None,
    i=0,
    free_color="b",
    fixed_color="c",
    surf_color=(0.1, 0.55, 0.85),
    surf_alpha=1.0,
):
    """
    Plot a 3D triangular mesh with free and fixed vertices.

    Parameters
    ----------
    V        : (n,3) array
               Free‐vertex coordinates.
    V_fixed  : (m,3) array
               Fixed‐vertex coordinates.
    T        : (k,3) int array
               Zero‐based face indices.
    fig      : matplotlib.figure.Figure, optional
    ax       : matplotlib.axes._subplots.Axes3DSubplot, optional
    free_color : color for free vertices
    fixed_color: color for fixed vertices
    surf_color : face color for triangulation
    surf_alpha : transparency of surface

    Returns
    -------
    fig, ax  : the figure and 3D axes (so you can reuse them)
    """
    # Create or clear figure + 3D axes
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax.cla()

    # Scatter free vs fixed vertices
    ax.scatter(
        V[:, 0], V[:, 1], V[:, 2], c=free_color, marker="o", label="Free Vertices"
    )
    ax.scatter(
        V_fixed[:, 0],
        V_fixed[:, 1],
        V_fixed[:, 2],
        c=fixed_color,
        marker="^",
        label="Fixed Vertices",
    )

    # build a triangulation object
    triang = mtri.Triangulation(V[:, 0], V[:, 1], triangles=T)

    # plot with plasma colormap
    surf = ax.plot_trisurf(
        triang,
        V[:, 2],
        cmap="plasma",
        edgecolor="k",
        linewidth=1.5,
        alpha=surf_alpha,
        vmin=0.0,  # lower color limit
        vmax=0.05,  # upper color limit
    )

    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    # Set limits based on vertex coordinates
    ax.set_xlim([np.min(V[:, 0]), np.max(V[:, 0])])
    ax.set_ylim([np.min(V[:, 1]), np.max(V[:, 1])])
    ax.set_zlim([-0.01, 0.5])

    # Labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Draw/update
    # plt.draw()
    # plt.pause(0.01)
    plt.savefig(f"out/plot{i}.png", dpi=300, bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    main()

# import os
# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from scipy.sparse import eye, kron
from scipy.sparse.linalg import LinearOperator

import scipy

import pyamg
import functools

import time

from numba import njit

import pyvista as pv

from libMobility import NBody, SelfMobility
import c_rigid_obj as cbodies
# import sksparse.cholmod


def main():
    in_dir = "in/"
    # file_prefix = "subd_6_"   ## change file prefix to use different membrane. subd_6 is the large membrane
    file_prefix = ""
    prefix = in_dir + file_prefix
    T = np.loadtxt(prefix + "faces.txt", dtype=int)
    T -= 1  # subtract 1 from T to convert from 1-based to 0-based indexing
    V_free = np.loadtxt(prefix + "free_vertex.txt", dtype=float)
    V_fixed = np.loadtxt(prefix + "fixed_vertex.txt", dtype=float)
    V_free[:, 2] = 0.0
    V_fixed[:, 2] = 0.0
    V = np.vstack((V_free, V_fixed))

    N_free = V_free.shape[0]
    N_fixed = V_fixed.shape[0]

    ### these are obviously the different sizes of rigid bodies.
    ### everything is set up to resize the membrane blobs to the size of the rigid blobs
    # shell_file = "structures/shell_N_12_Rg_0_7921_Rh_1.vertex"
    shell_file = "structures/shell_N_42_Rg_0_8913_Rh_1.vertex"
    # shell_file = "structures/shell_N_162_Rg_0_9497_Rh_1.vertex"
    rigid_sep, rigid_cfg = load_rigid_data(shell_file)
    rigid_radius = 1.0

    rigid_scale = 1.0 ### this is somewhat untested but should scale the rigid particle to be larger (and handle membrane scaling as well)
    rigid_cfg *= rigid_scale  # scale the rigid configuration
    rigid_radius *= rigid_scale  # scale the rigid radius

    N_bodies = 9  # number of rigid bodies requested
                  ### note that fewer rigid bodies may be generated if the configuration is too dense
                  ### the actual number is generated (and returned) by generate_rigid_sphere_config below.

    rigid_X0, N_bodies = generate_rigid_sphere_config(N_bodies, rigid_radius)
    # print(np.min(scipy.spatial.distance.pdist(rigid_X0)))
    rigid_X0[:, 2] += 2.0 * rigid_radius  # place the rigid bodies above the membrane
    print("Rigid body center heights:")
    print(np.min(rigid_X0[:, 2]), np.max(rigid_X0[:, 2]))
    rigid_X0 = rigid_X0.flatten()

    N_rigid = rigid_cfg.shape[0] * N_bodies

    Quat = np.array([1.0, 0.0, 0.0, 0.0] * N_bodies)

    N = N_free + N_fixed + N_rigid

    tri_nbs = get_diamonds(T, V)

    # find mesh size as the min distance between two vertices
    # mesh_size = np.inf
    # for i in range(V.shape[0]):
    #     for j in range(i + 1, V.shape[0]):
    #         dist = np.linalg.norm(V[i] - V[j])
    #         if dist < mesh_size:
    #             mesh_size = dist

    if file_prefix == "subd_6_":
        mesh_size = 0.01562499999999984
    elif file_prefix == "":
        mesh_size = 0.06249999999999988
    else:
        raise ValueError("Unknown file prefix for mesh size calculation")

    print(f"Mesh size: {mesh_size}")
    print(
        f"Number of free vertices: {N_free}, fixed vertices: {N_fixed}, rigid bodies: {N_rigid}"
    )

    # scale mesh so the blob size is the same as the rigid blobs
    scale_factor = rigid_sep / mesh_size
    V *= scale_factor
    V_free *= scale_factor
    V_fixed *= scale_factor
    mesh_size *= scale_factor

    print(f"Scaled mesh size: {mesh_size}")
    print("Rigid sep size:", rigid_sep)

    blob_radius = 0.5 * mesh_size  # radius of the blob for the mobility solver

    kbt = 0.0
    eta = 1.0
    k_bend = 1000
    k_spring = 0.0

    alpha = 1.0
    m0 = 1.0 / (6 * np.pi * eta * blob_radius)  # mobility coefficient
    dt = alpha * (mesh_size**3) / (m0 * k_bend)
    print("DT:", dt)

    solver = NBody("open", "open", "open")
    solver.setParameters()
    solver.initialize(viscosity=eta, hydrodynamicRadius=blob_radius)

    # pc_solver = SelfMobility("open", "open", "open")
    # pc_solver.initialize(temperature=kbt, viscosity=eta, hydrodynamicRadius=hydro_radius)

    # scale rigid configuration to have the same blob size as the membrane
    # rigid_cfg *= 2 * blob_radius / rigid_sep
    # rigid_radius *= 2 * blob_radius / rigid_sep

    cb = cbodies.CManyBodies()
    cb.setBlkPC(False)
    cb.setWallPC(False)
    cb.setParameters(N_rigid, blob_radius, dt, kbt, eta, np.array([0, 0, 0]), rigid_cfg)

    cb.setConfig(rigid_X0, Quat)
    cb.set_K_mats()

    rigid_Xn = rigid_X0.copy()

    # ranges for different indices
    full_size = 3 * (N + N_free) + 6 * N_bodies
    rigid_range = slice(0, 3 * N_rigid)
    free_range = slice(3 * N_rigid, 3 * (N_rigid + N_free))
    u_rigid_range = slice(3 * N, 3 * N + 6 * N_bodies)
    u_free_range = slice(3 * N + 6 * N_bodies, full_size)

    # plot_mesh(V_free, V_fixed, T, rigid_X0)

    final_time = 20.0  # final time
    Nsteps = int(final_time / dt)  # number of time steps

    n_plot = 10
    n_report = 5
    fig_count = 0
    for step in range(Nsteps):
        # --- Implicit update: solve for dX --- # TODO outdated, update the equation
        #    ( I + dt*Kx ) dX = -dt * ( Kx * X_free + RHSf - F_push )

        #### build membrane forces
        # bending force
        row, col, val, bending_boundary_contribution = assemble_willmore(
            tri_nbs, V, N_free
        )
        D = coo_matrix((val, (row, col)), shape=(N_free, N_free)).tocsr()
        D *= k_bend
        bending_boundary_contribution *= k_bend
        Dx = kron(
            D, eye(3, format="csr")
        )  # D is (Nfree×Nfree); Dx will be (3*Nfree×3*Nfree)

        # bending force
        X0 = V.flatten()
        F_bend = Dx.dot(X0[: 3 * N_free])

        # inhomogeneous boundary contribution from bending
        bending_bc_free = bending_boundary_contribution[
            : 3 * N_free
        ].copy()  # Truncate RHS to the free‐DOFs only:

        F_push = np.zeros(3 * N_free, dtype=float)
        # F_push[2] = -10.0 * max(0.0, 10.0 - dt * step)

        membrane_forces = F_bend + bending_bc_free + F_push

        freq = 1
        rigid_force_torque = np.zeros(6 * N_bodies, dtype=float)
        rigid_force_torque[4::6] = (  # roller torque
            8 * np.pi * eta * rigid_radius**3 * (2 * np.pi * freq)
        )

        for rb in range(N_bodies):
            rb_i = rb * 3
            spring_force = k_spring * (
                rigid_X0[rb_i : rb_i + 3] - rigid_Xn[rb_i : rb_i + 3]
            )
            rigid_force_torque[rb * 6 : rb * 6 + 3] = spring_force

        start = time.time()

        Bend_mat = dt * Dx

        mob_coeff = 1.0 / (6 * np.pi * eta * blob_radius)
        PC_mat = make_sparse_PC_mat(
            mob_coeff, cb, Bend_mat, N_rigid, N_free, N_fixed
        )

        # cond_num = np.linalg.cond(PC_mat.toarray())
        # print(f"Condition number: {cond_num:.2e}")
        # u, s, vt = scipy.sparse.linalg.svds(PC_mat, ak=10)
        # print("Smallest singular value:", s.min())


        # import matplotlib.pyplot as plt
        # plt.spy(PC_mat, markersize=0.5)
        # plt.title("Sparsity pattern of PC_mat")
        # plt.savefig("out/PC_mat_spy.png")
        # plt.close()
        # exit()

        # PC_decomp = sksparse.cholmod.cholesky(PC_mat)
        PC_decomp = scipy.sparse.linalg.splu(PC_mat, permc_spec="COLAMD")

        # combine V and rigid_pos into a single array
        rigid_pos = np.array(cb.multi_body_pos())
        rigid_pos = rigid_pos.reshape((-1, 3))
        all_pos = np.vstack((rigid_pos,V))
        solver.setPositions(all_pos)

        RHS = np.zeros(3 * N + 6 * N_bodies + 3 * N_free, dtype=float)
        RHS[u_rigid_range] = -rigid_force_torque
        RHS[u_free_range] = -membrane_forces

        A_partial = functools.partial(
            apply_A,
            N=N,
            solver=solver,
            cb=cb,
            Bend_mat=Bend_mat,
            rigid_range=rigid_range,
            free_range=free_range,
            u_rigid_range=u_rigid_range,
            u_free_range=u_free_range,
        )

        PC = LinearOperator(
            (full_size, full_size),
            # lambda x: PC_decomp(x),
            PC_decomp.solve,
            dtype=np.float64,
        )

        A = LinearOperator(
            shape=(full_size, full_size), matvec=A_partial, dtype=np.float64
        )

        res = []
        sol, info = pyamg.krylov.fgmres(
            A, RHS, tol=1e-4, x0=None, restart=100, maxiter=1000, residuals=res, M=PC
        )
        if step % n_report == 0:
            print("membrane heights:", np.min(V[:, 2]), np.max(V[:, 2]))
            print(f"GMRES converged in {len(res)} iters, residuals: {res[-1]}")
        # sol, info = gmres(A, RHS, rtol=1e-4, x0=None, restart=100, maxiter=1000)

        u_free = sol[u_free_range]
        end = time.time()
        print(f"Solving took {end - start:.4f} seconds")

        U_rigid = sol[u_rigid_range]
        cb.evolve_X_Q(U_rigid)
        _, rigid_Xn = cb.getConfig()

        # Compute new X
        X1 = X0.copy()
        X1[: 3 * N_free] = X0[: 3 * N_free] + dt * u_free

        V = X1.reshape((-1, 3))
        if step % n_plot == 0:
            print(f"Step {step}, plotting")
            plot_mesh(V[0:N_free, :], V[N_free:, :], T, rigid_Xn, i=fig_count)
            fig_count += 1


def generate_rigid_sphere_config(N, R, max_attempts=10000):
    seed = 10
    np.random.seed(seed)

    big_radius = 4 * R
    centers = []

    def is_valid(candidate, existing_centers):
        for c in existing_centers:
            if np.linalg.norm(candidate - c) < 2.3 * R:
                return False
        return True

    attempts = 0
    while len(centers) < N and attempts < max_attempts:
        # Sample a random point uniformly within a sphere of radius 4R
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = (np.random.rand() ** (1 / 3)) * (
            big_radius - R
        )  # Leave space for full radius
        candidate = direction * radius

        if is_valid(candidate, centers):
            centers.append(candidate)

        attempts += 1

    if len(centers) < N:
        print(
            f"Warning: Only placed {len(centers)} out of {N} spheres after {max_attempts} attempts."
        )

    return np.array(centers), len(centers)


def make_sparse_PC_mat(mob_coeff, cb, Bend_mat, N_rigid, N_membrane, N_fix, eps=1e-3):
    M_rigid = scipy.sparse.diags_array(
        [mob_coeff], offsets=0, shape=(3 * N_rigid, 3 * N_rigid), format="csc"
    )
    M_membrane = scipy.sparse.diags_array(
        [mob_coeff], offsets=0, shape=(3 * N_membrane, 3 * N_membrane), format="csc"
    )
    M_fix = scipy.sparse.diags_array(
        [mob_coeff], offsets=0, shape=(3 * N_fix, 3 * N_fix), format="csc"
    )
    K, _ = cb.get_K_Kinv()

    I_membrane = eye(3*N_membrane, 3*N_membrane, format="csc")

    b = scipy.sparse.block_array(
        [
            [M_rigid, None, None, -K, None],
            [None, M_membrane, None, None, -I_membrane],
            [None, None, M_fix, None, None],
            [-K.T, None, None, None, None],
            [None, I_membrane, None, None, Bend_mat],
        ]
    )

    I_eps = scipy.sparse.diags_array(
        [eps], offsets=0, shape=(b.shape[0], b.shape[0]), format="csc"
    )
    # b += I_eps

    return csc_matrix(b) # this cast shouldn't be needed but cholmod is a fuck

def apply_A(vec, N, solver, cb, Bend_mat, rigid_range, free_range, u_rigid_range, u_free_range):
    vec = np.array(vec, dtype=float)

    lam = vec[0 : 3 * N]
    Mf, _ = solver.Mdot(forces=lam)

    u_rigid = vec[u_rigid_range]
    Ku_rigid = cb.K_x_U(u_rigid)

    u_free = vec[u_free_range]

    lambda_rigid = vec[rigid_range]
    KT_lam_rigid = cb.KT_x_Lam(lambda_rigid)

    lambda_free = vec[free_range]

    out = np.zeros_like(vec)

    U = np.zeros_like(lam)
    U[rigid_range] = Ku_rigid
    U[free_range] = u_free

    out[0 : 3 * N] = Mf - U
    out[u_rigid_range] = -KT_lam_rigid
    out[u_free_range] = lambda_free + Bend_mat.dot(u_free)

    return out


def load_rigid_data(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        # s should be a float
        s = float(lines[0].split()[1])
        Cfg = np.array([[float(j) for j in i.split()] for i in lines[1:]])
    return s, Cfg


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


def plot_mesh(
    V_free,
    V_fixed,
    T,
    rigid_centers,
    i=0,
    free_color="blue",
    fixed_color="cyan",
    surf_color=(0.1, 0.55, 0.85),
    surf_alpha=1.0,
):
    """
    Plot a 3D triangular mesh with free and fixed vertices using PyVista.

    Parameters
    ----------
    V_free : (n,3) array
        Free vertex coordinates.
    V_fixed : (m,3) array
        Fixed vertex coordinates.
    T : (k,3) int array
        Triangle indices.
    rigid_cfg : (r,3) array
        Rigid body vertex positions.
    i : int
        Frame index for saving image.
    free_color : str or RGB tuple
        Color for free vertices.
    fixed_color : str or RGB tuple
        Color for fixed vertices.
    surf_color : RGB tuple
        Color for the mesh surface.
    surf_alpha : float
        Opacity of the surface.

    Returns
    -------
    plotter : pyvista.Plotter
        The PyVista plotter object.
    """

    # Combine vertices
    V = np.vstack((V_free, V_fixed))

    # Create a mesh
    mesh = pv.PolyData(V, np.hstack((np.full((T.shape[0], 1), 3), T)).astype(np.int_))

    # Start plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")

    # Add surface mesh
    plotter.add_mesh(
        mesh,
        color=surf_color,
        opacity=surf_alpha,
        show_edges=False,
        scalars=V[:, 2],  # Color by Z value
        cmap="turbo",
        clim=[-0.8, 0.8],
    )

    # Add free vertices
    # plotter.add_points(
    #     V_free,
    #     color=free_color,
    #     point_size=8,
    #     render_points_as_spheres=True,
    #     label="Free Vertices",
    # )

    # Add fixed vertices
    plotter.add_points(
        V_fixed,
        color=fixed_color,
        point_size=8,
        render_points_as_spheres=True,
        label="Fixed Vertices",
    )

    R = 1.0
    rigid_centers = rigid_centers.reshape((-1, 3))
    center_ind = 0
    for center in rigid_centers:
        sphere = pv.Sphere(
            radius=R, center=center, theta_resolution=30, phi_resolution=30
        )

        if center_ind == 0:
            color = "darkslateblue"
        else:
            color = "skyblue"
        center_ind += 1

        plotter.add_mesh(sphere, color=color, smooth_shading=True)

    # Add legend and axes
    plotter.add_legend()
    plotter.show_axes()

    # plotter.camera_position = [
    #     (20, -20, 8),  # camera location
    #     (0, 0, 0),  # focal point
    #     (0, 0, 1),  # view up direction
    # ]

    plotter.camera_position = [
        (0, -40, 20),  # camera location
        (0, 0, 8),  # focal point
        (0, 0, 1),  # view up direction
    ]

    # Save the figure to file
    plotter.screenshot(f"out/plot{i}.png")
    plotter.close()


if __name__ == "__main__":
    main()

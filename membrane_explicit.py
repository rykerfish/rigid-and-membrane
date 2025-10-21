# import os
# os.environ["OMP_NUM_THREADS"] = "1"
import json
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse import eye, kron
from scipy.sparse.linalg import LinearOperator
import scipy
import pyamg
import functools
import time
from numba import njit, prange
import pyvista as pv
from libMobility import NBody, SelfMobility
from Rigid import RigidBody


def main():
    k_bend = 500.0
    run(k_bend)


def run(k_bend):

    in_dir = "in/"
    save_dir = "out/explicit/"
    file_prefix = "subd_6_"  ## change file prefix to use different membrane. subd_6 is the large membrane
    # file_prefix = ""
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
    ### everything is set up to resize the membrane blobs to the sise of the rigid blobs
    # shell_file = "structures/shell_N_12_Rg_0_7921_Rh_1.vertex"
    shell_file = "structures/shell_N_42_Rg_0_8913_Rh_1.vertex"
    # shell_file = "structures/shell_N_162_Rg_0_9497_Rh_1.vertex"
    rigid_sep, rigid_cfg = load_rigid_data(shell_file)
    rigid_radius = 1.0

    rigid_scale = 1.0  ### this is somewhat untested but should scale the rigid particle to be larger (and handle membrane scaling as well)
    rigid_cfg *= rigid_scale  # scale the rigid configuration
    rigid_radius *= rigid_scale  # scale the rigid radius

    N_bodies = 9  # number of rigid bodies requested
    ### note that fewer rigid bodies may be generated if the configuration is too dense
    ### the actual number is generated (and returned) by generate_rigid_sphere_config below.

    rigid_X0, N_bodies = generate_rigid_sphere_config(N_bodies, rigid_radius)
    print(
        f"Generated {N_bodies} rigid bodies with radius {rigid_radius} and separation {rigid_sep}"
    )

    rigid_X0[:, 2] += 5.0 * rigid_radius  # place the rigid bodies above the membrane
    print("Rigid body center heights:")
    print(np.min(rigid_X0[:, 2]), np.max(rigid_X0[:, 2]))
    print("min rigid sep: ", np.min(scipy.spatial.distance.pdist(rigid_X0)))
    rigid_X0 = rigid_X0.flatten()

    N_rigid = rigid_cfg.shape[0] * N_bodies

    quat = np.repeat(np.array([1.0, 0.0, 0.0, 0.0]), N_bodies, axis=0)

    N = N_free + N_fixed + N_rigid

    tri_nbs = get_diamonds(T, V)
    one_rings = get_one_rings(N_free + N_fixed, T)

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

    eta = 1.0

    # alpha = 1e-1
    alpha = 1e-1
    m0 = 1.0 / (6 * np.pi * eta * blob_radius)  # mobility coefficient
    dt = alpha * (mesh_size**3) / (m0 * k_bend)
    rest_length = 2 * blob_radius
    k_spring = 50.0
    mg = 5.0

    print("dt:", dt)

    solver = NBody("open", "open", "open")
    solver.setParameters()
    solver.initialize(viscosity=eta, hydrodynamicRadius=blob_radius)

    cb = RigidBody(
        rigid_config=rigid_cfg, X=rigid_X0, Q=quat, a=blob_radius, eta=eta, dt=dt
    )

    rigid_Xn = rigid_X0.copy()

    # ranges for different indices
    full_size = 3 * (N + N_free) + 6 * N_bodies
    rigid_range = slice(0, 3 * N_rigid)
    free_range = slice(3 * N_rigid, 3 * (N_rigid + N_free))
    u_rigid_range = slice(3 * N, 3 * N + 6 * N_bodies)
    u_free_range = slice(3 * N + 6 * N_bodies, full_size)

    final_time = 20.0  # final time
    N_steps = int(final_time / dt)  # number of time steps
    # N_steps = 10000001
    print(f"Number of steps: {N_steps}")
    # print(f"Final time: {final_time}, dt: {dt}")

    blob_fname = save_dir + "blob_pos.csv"
    rigid_fname = save_dir + "rigid_pos.csv"
    print(f"saveing to {save_dir}")

    blob_params = {
        "rigid_radius": rigid_radius,
        "blob_radius": blob_radius,
        "N_spheres": N_bodies,
        "N_blobs_per_sphere": rigid_cfg.shape[0],
        "N_free": N_free,
        "N_fixed": N_fixed,
        "eta": eta,
        "dt": dt,
        "k_bend": k_bend,
        "mesh_size": mesh_size,
        "rigid_sep": rigid_sep,
    }
    json.dump(blob_params, open(save_dir + "params.json", "w"), indent=4)

    n_plot = 1
    n_save = 200
    # n_save_streamline = 5000
    # n_save_velocity = 100
    n_report = 1
    fig_count = 0
    for step in range(N_steps):
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
        X0 = V.flatten()
        F_bend = Dx.dot(X0[: 3 * N_free])

        # inhomogeneous boundary contribution from bending
        bending_bc_free = bending_boundary_contribution[
            : 3 * N_free
        ].copy()  # Truncate RHS to the free‐DOFs only:

        F_push = np.zeros(3 * N_free, dtype=float)
        F_spring = edge_springs(V, T, one_rings, k_spring, rest_length)
        F_push -= F_spring.T.ravel(order="F")[: 3 * N_free]

        membrane_forces = F_bend + bending_bc_free + F_push

        freq = 3
        rigid_force_torque = np.zeros(6 * N_bodies, dtype=float)
        rigid_force_torque[4::6] = (  # roller torque
            8 * np.pi * eta * rigid_radius**3 * (2 * np.pi * freq)
        )
        rigid_force_torque[2::6] = -mg  # gravity force

        start = time.time()

        mob_coeff = 1.0 / (6 * np.pi * eta * blob_radius)
        K = cb.get_K()
        PC_mat = make_sparse_PC_mat(mob_coeff, K, N_rigid, N_free, N_fixed)
        PC_decomp = scipy.sparse.linalg.splu(PC_mat, permc_spec="COLAMD")

        # combine V and rigid_pos into a single array
        rigid_pos = np.array(cb.get_blob_positions())
        rigid_pos = rigid_pos.reshape((-1, 3))
        all_pos = np.vstack((rigid_pos, V))
        solver.setPositions(all_pos)

        RHS = np.zeros(3 * N + 6 * N_bodies + 3 * N_free, dtype=float)
        RHS[u_rigid_range] = -rigid_force_torque
        RHS[u_free_range] = membrane_forces

        A_partial = functools.partial(
            apply_A,
            N=N,
            solver=solver,
            cb=cb,
            rigid_range=rigid_range,
            free_range=free_range,
            u_rigid_range=u_rigid_range,
            u_free_range=u_free_range,
        )

        PC = LinearOperator(
            (full_size, full_size),
            # lambda x: PC_decomp(x),
            PC_decomp.solve,
            dtype=cb.precision,
        )

        A = LinearOperator(
            shape=(full_size, full_size), matvec=A_partial, dtype=cb.precision
        )

        norm_rhs = np.linalg.norm(RHS)
        # print(f"Norm of RHS: {norm_rhs:.4f}")
        # print("norm of A@RHS", np.linalg.norm(A.dot(RHS / norm_rhs)))

        res = []
        sol, info = pyamg.krylov.fgmres(
            A,
            RHS / norm_rhs,
            tol=1e-2,
            x0=None,
            restart=100,
            maxiter=1000,
            residuals=res,
            M=PC,
        )
        if step % n_report == 0:
            print("membrane heights:", np.min(V[:, 2]), np.max(V[:, 2]))
            print(f"GMRES converged in {len(res)} iters, residuals: {res[-1]}")

        sol *= norm_rhs

        u_free = sol[u_free_range]
        end = time.time()
        # print(f"Solving took {end - start:.4f} seconds")

        U_rigid = sol[u_rigid_range]

        cb.evolve_rigid_bodies(U_rigid)

        rigid_Xn, _ = cb.get_config()

        # Compute new X
        X1 = X0.copy()
        X1[: 3 * N_free] = X0[: 3 * N_free] + dt * u_free

        V = X1.reshape((-1, 3))
        if step % n_plot == 0:
            print(f"Step {step}, plotting")

            # np.savetxt(save_dir + f"U_rigid_{step}.txt", U_rigid)
            # plot_mesh(
            #     V[0:N_free, :],
            #     V[N_free:, :],
            #     T,
            #     rigid_Xn,
            #     u_streamlines=u_streamlines,
            #     pv_mesh=mesh,
            #     i=fig_count,
            # )

            plot_mesh(
                V[0:N_free, :],
                V[N_free:, :],
                T,
                rigid_Xn,
                u_streamlines=None,
                pv_mesh=None,
                save_dir=save_dir,
                i=fig_count,
            )
            fig_count += 1

        if step % n_save == 0:
            with open(blob_fname, "a") as f:
                np.savetxt(
                    f, np.reshape(all_pos.flatten(), (1, -1)), delimiter=",", fmt="%.6f"
                )
            with open(rigid_fname, "a") as f:
                np.savetxt(
                    f,
                    np.reshape(rigid_Xn.flatten(), (1, -1)),
                    delimiter=",",
                    fmt="%.6f",
                )

        # if step % n_save_streamline == 0:
        #     print("saving streamlines and mesh")
        #     nx = 51
        #     ny = 51
        #     nz = 51
        #     origin = [-5.0, -5.0, 0.5]
        #     s = 0.2
        #     spacing = (s, s, s)
        #     mesh = pv.ImageData(dimensions=(nx, ny, nz), spacing=spacing, origin=origin)

        #     x = mesh.points[:, 0]
        #     y = mesh.points[:, 1]
        #     z = mesh.points[:, 2]
        #     cube_pts = np.column_stack((x, y, z))

        #     all_hydro_points = np.vstack((rigid_pos, V, cube_pts))
        #     solver.setPositions(all_hydro_points)

        #     # make a single vector out of (sol[0:full_size], np.zeros(cube_pts.shape[0] * 3))
        #     F_streamlines = np.concatenate(
        #         [sol[0 : 3 * N], np.zeros(cube_pts.shape[0] * 3, dtype=sol.dtype)]
        #     )
        #     u_streamlines, _ = solver.Mdot(forces=F_streamlines)
        #     u_streamlines = u_streamlines[3 * N :].reshape((-1, 3))

        #     np.savetxt(save_dir + f"u_streamlines_{step}.txt", u_streamlines)
        #     np.savetxt(save_dir + f"V_{step}.txt", V)
        #     np.savetxt(save_dir + f"rigid_Xn_{step}.txt", rigid_Xn)
        #     np.savetxt(save_dir + f"streamline_mesh_{step}.txt", cube_pts)

        # if step % n_save_velocity == 0:
        #     with open(save_dir + f"U_rigid_mean.txt", "a") as f:
        #         U_rigid_tmp = U_rigid.reshape((N_bodies, 6))
        #         avg_U_rigid = np.mean(U_rigid_tmp, axis=0)
        #         current_deflection = np.max(V[:, 2]) - np.min(V[:, 2])
        #         f.write(
        #             f"{dt*step} {avg_U_rigid[0]} {avg_U_rigid[1]} {avg_U_rigid[2]} {avg_U_rigid[3]} {avg_U_rigid[4]} {avg_U_rigid[5]} {current_deflection}\n"
        #         )


def generate_rigid_sphere_config(N, R, max_attempts=10000):
    seed = 10
    np.random.seed(seed)

    big_radius = 5 * R
    centers = []

    def is_valid(candidate, existing_centers):
        for c in existing_centers:
            if np.linalg.norm(candidate - c) < 2.5 * R:
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


def make_sparse_PC_mat(mob_coeff, K, N_rigid, N_membrane, N_fix):
    M_rigid = scipy.sparse.diags_array(
        [mob_coeff], offsets=0, shape=(3 * N_rigid, 3 * N_rigid), format="csc"
    )
    M_membrane = scipy.sparse.diags_array(
        [mob_coeff], offsets=0, shape=(3 * N_membrane, 3 * N_membrane), format="csc"
    )
    M_fix = scipy.sparse.diags_array(
        [mob_coeff], offsets=0, shape=(3 * N_fix, 3 * N_fix), format="csc"
    )

    I_membrane = eye(3 * N_membrane, 3 * N_membrane, format="csc")

    b = scipy.sparse.block_array(
        [
            [M_rigid, None, None, -K, None],
            [None, M_membrane, None, None, -I_membrane],
            [None, None, M_fix, None, None],
            [-K.T, None, None, None, None],
            [None, -I_membrane, None, None, None],
        ]
    )

    return csc_matrix(b)


def apply_A(vec, N, solver, cb, rigid_range, free_range, u_rigid_range, u_free_range):
    vec = np.array(vec, dtype=float)

    lam = vec[0 : 3 * N]
    Mf, _ = solver.Mdot(forces=lam)

    u_rigid = vec[u_rigid_range]
    # Ku_rigid = K @ u_rigid
    Ku_rigid = cb.K_dot(u_rigid)

    u_free = vec[u_free_range]

    lambda_rigid = vec[rigid_range]
    # KT_lam_rigid = K.T @ lambda_rigid
    KT_lam_rigid = cb.KT_dot(lambda_rigid)

    lambda_free = vec[free_range]

    out = np.zeros_like(vec)

    U = np.zeros_like(lam)
    U[rigid_range] = Ku_rigid
    U[free_range] = u_free

    out[0 : 3 * N] = Mf - U
    out[u_rigid_range] = -KT_lam_rigid
    out[u_free_range] = -(lambda_free)

    return out


def load_rigid_data(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
        # s should be a float
        s = float(lines[0].split()[1])
        Cfg = np.array([[float(j) for j in i.split()] for i in lines[1:]])
    return s, Cfg


# @jit(parallel=True, fastmath=True)
def get_one_rings(numv, faces):
    # return vertices of one ring
    one_ring = [set() for index in range(numv)]
    for j in range(len(faces)):
        t = faces[j]
        one_ring[t[0]].update([t[1]])
        one_ring[t[0]].update([t[2]])
        one_ring[t[1]].update([t[0]])
        one_ring[t[1]].update([t[2]])
        one_ring[t[2]].update([t[0]])
        one_ring[t[2]].update([t[1]])

    one_ring = [np.array(list(OR), np.int32) for OR in one_ring]
    return one_ring


@njit(parallel=True, fastmath=True)
def edge_springs(vertices, faces, one_rings, k_spring, rest_l):
    # return spring force
    numv = vertices.shape[0]
    numt = faces.shape[0]
    force_springs = np.zeros((numv, 3))
    for i in prange(numv):
        j = 0
        for k in one_rings[i]:
            E_k = vertices[k, :] - vertices[i, :]
            E_k_n = np.linalg.norm(E_k)
            # Hookean spring
            force_springs[i, :] += (
                1.0 * k_spring * (1.0 - (rest_l / E_k_n)) * E_k
            )  # rest_l[i,j]
            # FENE spring
            # force_springs[i,:] += 1.0 * k_spring / (1.0 - (E_k_n/rest_l)**2) * E_k
            j += 1
    return force_springs


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
    u_streamlines=None,
    pv_mesh=None,
    i=0,
    save_dir="out/",
    surf_alpha=1.0,
):

    teal = np.array([67, 179, 174]) / 255.0
    yellow = np.array([255, 133, 0]) / 255.0
    pank = np.array([255, 0, 144]) / 255.0
    pants = np.array([65, 105, 225]) / 255.0

    # Combine vertices
    V = np.vstack((V_free, V_fixed))

    # Create a mesh
    mesh = pv.PolyData(V, np.hstack((np.full((T.shape[0], 1), 3), T)).astype(np.int_))

    # Start plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")

    # Directional light (simulates sunlight)
    light1 = pv.Light(light_type="headlight")
    light1.intensity = 0.8
    plotter.add_light(light1)

    plotter.enable_shadows()
    # plotter.shadow_map_resolution = 4096

    # Add surface mesh
    plotter.add_mesh(
        mesh,
        opacity=surf_alpha,
        show_edges=False,
        scalars=V[:, 2],  # Color by Z value
        cmap="balance",
        clim=[-3.0, 3.0],
        smooth_shading=True,
        specular=0.5,
        specular_power=25,
    )

    R = 1.0
    rigid_centers = rigid_centers.reshape((-1, 3))
    for center in rigid_centers:
        sphere = pv.Sphere(
            radius=R, center=center, theta_resolution=128, phi_resolution=128
        )

        plotter.add_mesh(
            sphere, color=yellow, smooth_shading=True, specular=0.5, specular_power=15
        )

    if u_streamlines is not None and pv_mesh is not None:
        pv_mesh["vectors"] = u_streamlines
        stream, src = pv_mesh.streamlines(
            "vectors",
            integration_direction="forward",
            return_source=True,
            terminal_speed=1e-2,
            n_points=150,
            source_radius=7.0,
            max_length=5.0,
        )
        plotter.add_mesh(stream.tube(radius=0.1))

    plotter.show_axes()

    plotter.camera_position = [
        (0, -30, 10),  # camera location
        (0, 0, 3),  # focal point
        (0, 0, 1),  # view up direction
    ]

    # Save the figure to file
    plotter.screenshot(save_dir + f"plot{i}.png")
    plotter.close()


if __name__ == "__main__":
    main()

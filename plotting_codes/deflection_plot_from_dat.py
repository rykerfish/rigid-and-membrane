from functools import partial
import json
import numpy as np
from plotoptix import TkOptiX
from plotoptix.materials import m_plastic, m_thin_walled, m_clear_glass
import cmasher
from matplotlib import cm

# from plotoptix.utils import map_to_colors  # feature to color conversion

yellow = np.array([255.0 / 255.0, 174.0 / 255.0, 26.0 / 255.0])
teal = np.array([67, 179, 174]) / 255.0
bbblue = np.array([176, 196, 222]) / 255.0
turq = np.array([0, 206, 209]) / 255.0
ly = np.array([250, 240, 190]) / 255.0
deeppeach = np.array([255, 203, 164]) / 255.0
bubs = np.array([231, 254, 255]) / 255.0
pblue = np.array([176, 224, 230]) / 255.0
thistle = np.array([216, 191, 216]) / 255.0
khaki = np.array([240, 230, 140]) / 255.0
blueb = np.array([182, 225, 242]) / 255.0
steel = np.array([242, 243, 244]) / 255.0
white = np.array([248, 248, 255]) / 255.0
pink = np.array([255, 0, 144]) / 255.0
silver = np.array([191, 193, 194]) / 255.0
silver2 = np.array([169, 169, 172]) / 255.0
dark_pink = np.array([139, 0, 139]) / 255.0
orange = np.array([255, 211, 0]) / 255.0
chart = np.array([127, 255, 0]) / 255.0
mint = np.array([0, 250, 154]) / 255.0
lime = np.array([50, 205, 50]) / 255.0
canary = np.array([255, 239, 0]) / 255.0
plat = np.array([229, 228, 226]) / 255.0
plat_cool = np.array([226, 228, 229]) / 255.0
jet = np.array([53, 56, 57]) / 255.0


class params:
    r_b = None
    tri_nbs = None
    V = None
    dX = None
    N_free = None
    dt = None
    k_bend = None
    cmap = None
    n = 0
    rigid_pos = None


def Advance_Particles(blob_dat_n):
    pass
    # row, col, val, RHS = assemble_willmore(params.tri_nbs, params.V, params.N_free)
    # val *= params.k_bend  # scale by the bending stiffness
    # RHS *= params.k_bend  # scale the RHS by the bending stiffness
    # K = coo_matrix((val, (row, col)), shape=(params.N_free, params.N_free)).tocsr()

    # # K is (Nfree×Nfree); Kx will be (3*Nfree×3*Nfree)
    # Kx = kron(K, eye(3, format="csr"))

    # # Truncate RHS to the free‐DOFs only:
    # RHSf = RHS[: 3 * params.N_free].copy()  # length = 3*Nfree

    # F_push = np.zeros(3 * params.N_free, dtype=float)  # no external forces
    # F_push[2] = (
    #     12.0 * params.k_bend * max(0.0, 1.0 - params.dt * params.n / 100.0)
    # )  # max(0.0,1.0-np.abs(params.dt*params.n)/100.0)

    # A = eye(3 * params.N_free, format="csr") + params.dt * Kx
    # X = params.V.T.ravel(order="F")
    # b = -params.dt * (Kx.dot(X[: 3 * params.N_free]) + RHSf + F_push)

    # # params.dX, info = gmres(A, b, rtol=1e-4, x0=params.dX, restart=100, maxiter=1000)
    # params.dX = b


# Perform any calculations here, including CPU-extensive work. But do
# not access ray tracing buffers, and do not update anything in the
# scene (geometry or plots data, lights, cameras, materials).
# Code in this function runs in parallel to the ray tracing.
def compute_changes(rt: TkOptiX, delta: int, blob_dat, N_rigid, N_membrane) -> None:
    blob_dat_n = blob_dat[params.n, :]
    params.rigid_pos = blob_dat_n[: 3 * N_rigid].reshape((N_rigid, 3))
    params.V = blob_dat_n[3 * N_rigid :].reshape((N_membrane, 3))
    params.n += 1
    # Advance_Particles(blob_dat_n)


# Access ray tracing buffers, modify/update scene, but do not launch
# time consuming work here as this will delay the next frame ray tracing.
# Code in this function runs synchronously with the ray tracing launches.
def update_scene(rt: TkOptiX, rigid_colors) -> None:
    # Compute new X
    # X = params.V.T.ravel(order="F")
    # Xnew = X.copy()
    # Xnew[: 3 * params.N_free] = X[: 3 * params.N_free] + params.dX
    # Reshape back into V:
    # params.V = Xnew.reshape((3, -1), order="F").T
    zz = np.copy(params.V[:, 2])
    # append -0.3 to the end of zz
    val = 0.5
    zz = np.concatenate(([val], zz, [-val]))
    cc = map_to_colors(zz, params.cmap)
    cc = cc[:-1, :]
    cc = cc[1:, :]
    rt.update_data("membrane", pos=params.V, c=cc)
    rt.set_data(
        "spheres", params.rigid_pos, r=params.r_b, c=rigid_colors, mat="plastic"
    )
    rt.update_graph("mesh", pos=params.V)
    rt.save_image("../img/optix/frame_{:05d}.png".format(params.n))


def face_2_edge(faces):
    edge = []
    for t in faces:
        edge.append([t[0], t[1]])
        edge.append([t[1], t[2]])
        edge.append([t[2], t[0]])
    return edge


def map_to_colors(x, cm_name: str) -> np.ndarray:
    """Map input variable to matplotlib color palette.

    Scale variable ``x`` to the range <0; 1> and map it to RGB colors from
    matplotlib's colormap with ``cm_name``.

    Parameters
    ----------
    x : array_like
        Input variable, array_like of any shape.
    cm_name : string
        Matplotlib colormap name.

    Returns
    -------
    out : np.ndarray
        Numpy array with RGB color values mapped from the input array values.
        The output shape is ``x.shape + (3,)``.
    """
    if x is None:
        raise ValueError()

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    min_x = x.min()
    max_x = x.max()
    if min_x != max_x:
        x = (1 / (x.max() - min_x)) * (x - min_x)
    else:
        x = np.zeros(x.shape)

    c = cm.get_cmap(cm_name)(x)
    return np.delete(c, np.s_[-1], len(c.shape) - 1)  # skip alpha


def main():

    data_dir = "../out/implicit/"
    fname = data_dir + "blob_pos.bin"
    bin_params = json.load(open(data_dir + "binary_metadata.json", "r"))
    blob_dat = np.fromfile(fname, bin_params["dtype"])
    blob_dat = np.reshape(blob_dat, (bin_params["n_rows"], bin_params["row_size"]))
    subsample_rate = 4
    blob_dat = blob_dat[0::subsample_rate, :]

    my_params = json.load(open(data_dir + "params.json", "r"))
    N_per = my_params["N_blobs_per_sphere"]
    N_rigid_blobs = N_per * my_params["N_spheres"]
    free_blobs = my_params["N_free"]
    fixed_blobs = my_params["N_fixed"]
    N_membrane_blobs = free_blobs + fixed_blobs

    # print max membrane z value
    membrane_dat = blob_dat[:, 3 * N_rigid_blobs :]
    min_membrane_z = np.min(membrane_dat[:, 2::3])
    max_membrane_z = np.max(membrane_dat[:, 2::3])
    print("min membrane z:", min_membrane_z)
    print("max membrane z:", max_membrane_z)

    big_mesh_size = 0.01562499999999984
    small_mesh_size = 0.06249999999999988
    scale_factor = 0.3 * big_mesh_size / small_mesh_size
    blob_dat *= scale_factor

    rigid_color = np.repeat(yellow, N_rigid_blobs)
    rigid_color = rigid_color.reshape(-1, 3, order="F")
    for i in range(0, N_rigid_blobs, N_per):
        rigid_color[i] = [0.0, 0.0, 0.0]

    update_scene_partial = partial(update_scene, rigid_colors=rigid_color)

    compute_partial = partial(
        compute_changes,
        blob_dat=blob_dat,
        N_rigid=N_rigid_blobs,
        N_membrane=N_membrane_blobs,
    )

    def print_cam(rt):
        print(rt.get_camera("cam1"))

    rt = TkOptiX(
        on_scene_compute=compute_partial,
        on_rt_accum_done=print_cam,
        on_rt_completed=update_scene_partial,
        width=2140,
        height=2140,
    )

    # 4 accumulation passes on each compute-update cycle:
    rt.set_param(min_accumulation_step=200)

    exposure = 0.5
    gamma = 2.2
    # rt.set_float("tonemap_exposure", exposure)
    # rt.set_float("tonemap_gamma", gamma)
    # rt.add_postproc("Gamma")
    rt.add_postproc("Denoiser")

    rt.setup_material("plastic", m_plastic)
    rt.setup_material("thin", m_thin_walled)
    rt.setup_material("glass", m_clear_glass)

    file_prefix = "../in/subd_6_"
    T = np.loadtxt(file_prefix + "faces.txt", dtype=int)
    T -= 1

    params.r_b = my_params["blob_radius"] * scale_factor
    params.k_bend = my_params["k_bend"]

    blob_dat_n = blob_dat[params.n, :]
    params.rigid_pos = blob_dat_n[: 3 * N_rigid_blobs].reshape((N_rigid_blobs, 3))
    params.V = blob_dat_n[3 * N_rigid_blobs :].reshape((N_membrane_blobs, 3))

    params.cmap = "cmr.viola"

    zz = np.copy(params.V[:, 2])
    val = 0.5
    zz = np.concatenate(([val], zz, [-val]))
    cc = map_to_colors(zz, params.cmap)
    cc = cc[:-1, :]
    cc = cc[1:, :]

    print(np.shape(params.V))
    print(np.shape(cc))

    rt.set_data("membrane", params.V, r=0.8 * params.r_b, c=cc, mat="plastic")
    rt.set_data(
        "spheres",
        params.rigid_pos,
        r=params.r_b,
        c=rigid_color,
        mat="plastic",
    )

    rt.set_graph("mesh", params.V, face_2_edge(T), c=0.15, r=0.1 * params.r_b)

    rt.set_ambient(0.1)
    rt.set_background(1)
    rt.set_float("scene_epsilon", 0.06)

    rt.setup_camera(
        "cam1",
        cam_type="DoF",
        eye=[0.8955327, 5.14466858, 0.6898626],
        target=[0.148875, 0.0106287, -0.07440066],
        up=[0, 0, 1],
        aperture_radius=0.02,  # 0.2,
        fov=25,
        focal_scale=0.99,
    )

    rt.setup_light("light1", pos=[-2, 2, 4], color=2, radius=1.1 * 2)
    rt.setup_light("light2", pos=[4, -2, 0], color=5, radius=1.1 * 1.5)
    rt.show()


if __name__ == "__main__":
    main()

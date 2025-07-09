import numpy as np
import pyvista as pv


def main():
    dir = "save/500/"
    suffix = "_20000.txt"
    V = np.loadtxt(dir + "V" + suffix)
    u_streamlines = np.loadtxt(dir + "u_streamlines" + suffix)
    # mesh = np.loadtxt(dir + 'streamline_mesh' + suffix)
    T = np.loadtxt("in/subd_6_faces.txt", dtype=int)
    T -= 1  # subtract 1 from T to convert from 1-based to 0-based indexing

    nx = 51
    ny = 51
    nz = 51
    origin = [-5.0, -5.0, 0.5]
    s = 0.2
    spacing = (s, s, s)
    mesh = pv.ImageData(dimensions=(nx, ny, nz), spacing=spacing, origin=origin)

    rigid_centers = np.loadtxt(dir + "rigid_Xn" + suffix, dtype=float)
    rigid_centers = rigid_centers.reshape((-1, 3))

    plot_mesh(V, T, rigid_centers, u_streamlines, mesh)


def plot_mesh(
    V,
    T,
    rigid_centers,
    u_streamlines=None,
    pv_mesh=None,
    surf_alpha=1.0,
):

    teal = np.array([67, 179, 174]) / 255.0
    yellow = np.array([255, 133, 0]) / 255.0
    pank = np.array([255, 0, 144]) / 255.0
    pants = np.array([65, 105, 225]) / 255.0

    # Create a mesh
    mesh = pv.PolyData(V, np.hstack((np.full((T.shape[0], 1), 3), T)).astype(np.int_))

    # window size can fix lighting bugs because the shadow_map is scuffed
    ws = 1500
    ratio = 1 / 1
    plotter = pv.Plotter(
        off_screen=True, window_size=(int(ratio * ws), ws), lighting="none"
    )
    plotter.set_background("white")
    # plotter.lights.clear() #

    # Directional light (simulates sunlight)
    # light1 = pv.Light(light_type="headlight")
    # light1.intensity = 0.8
    # plotter.add_light(light1)

    light2 = pv.Light(light_type="scene light")
    light2.intensity = 2.3
    light2.position = (0, -30, 15)
    light2.focal_point = (10, 0, 0)
    light2.positional = False
    plotter.add_light(light2)

    plotter.enable_shadows()
    # plotter.shadow_map_resolution = 4096

    cbar_args = dict(
        title="Membrane height",
        vertical=True,
        position_x=0.05,
        position_y=0.2,
        title_font_size=50,
        label_font_size=40,
    )

    # Add surface mesh
    g = 0.999
    mesh["z_scalar"] = V[:, 2]
    plotter.add_mesh(
        mesh,
        # opacity=surf_alpha,
        scalars="z_scalar",
        cmap="balance",
        clim=[-2.0, 2.0],
        # smooth_shading=True,
        specular=0.9,
        diffuse=1.0,
        specular_power=90,
        scalar_bar_args=cbar_args,
        line_width=0.5,
        edge_color=(g, g, g),
    )
    # plotter.remove_scalar_bar()

    R = 0.8
    rigid_centers = rigid_centers.reshape((-1, 3))
    for center in rigid_centers:
        sphere = pv.Sphere(
            radius=R, center=center, theta_resolution=128, phi_resolution=128
        )

        plotter.add_mesh(
            sphere, color=yellow, smooth_shading=True, specular=0.5, specular_power=15
        )

    if u_streamlines is not None and pv_mesh is not None:

        cbar_args = dict(
            title="Streamline velocity",
            vertical=True,
            position_x=0.1,
            position_y=0.2,
            title_font_size=50,
            label_font_size=40,
        )

        normalize_fact = 0.6597939552851914
        pv_mesh["vectors"] = u_streamlines / normalize_fact

        stream, src = pv_mesh.streamlines(
            "vectors",
            integration_direction="forward",
            return_source=True,
            terminal_speed=3,
            n_points=150,
            source_radius=5.0,
            source_center=(0, -5, 6),
            max_length=7.0,
        )
        plotter.add_mesh(stream.tube(radius=0.05), scalar_bar_args=cbar_args)
        # plotter.remove_scalar_bar("vectors")

    plotter.camera_position = [
        (0, -30, 8),  # camera location
        (0, 0, 10),  # focal point
        (0, 0, 1),  # view up direction
    ]

    # Save the figure to file
    plotter.screenshot(f"streamlines.png")
    plotter.close()


if __name__ == "__main__":
    main()

import numpy as np
import plotly.graph_objects as go

# Parameters
layout_type = "hexagonal"  # Options: square, circular, hexagonal, annular, radial, random
num_minibeams_per_side = 10
minibeam_spacing = 3.0        # mm center-to-center spacing
protons_per_minibeam = 1000
minibeam_width = 0.5          # mm (standard deviation for Gaussian spread)
initial_z = -150.0            # mm (Z-plane of initial beam)

minibeam_centers = []

if layout_type == "square":
    x_vals = np.linspace(-minibeam_spacing * (num_minibeams_per_side - 1) / 2,
                         minibeam_spacing * (num_minibeams_per_side - 1) / 2,
                         num_minibeams_per_side)
    y_vals = np.linspace(-minibeam_spacing * (num_minibeams_per_side - 1) / 2,
                         minibeam_spacing * (num_minibeams_per_side - 1) / 2,
                         num_minibeams_per_side)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    minibeam_centers = np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)

elif layout_type == "circular":
    radius = minibeam_spacing * num_minibeams_per_side / 2
    for i in range(num_minibeams_per_side**2):
        angle = np.random.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1))
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        minibeam_centers.append([x, y])

elif layout_type == "hexagonal":
    for i in range(num_minibeams_per_side):
        for j in range(num_minibeams_per_side):
            x = i * minibeam_spacing
            y = j * minibeam_spacing * np.sqrt(3)/2
            if j % 2 == 1:
                x += minibeam_spacing / 2
            minibeam_centers.append([x, y])
    minibeam_centers = np.array(minibeam_centers)
    center_offset = np.mean(minibeam_centers, axis=0)
    minibeam_centers -= center_offset

elif layout_type == "annular":
    rings = num_minibeams_per_side // 2
    for r in range(1, rings + 1):
        num_beams = 6 * r
        for i in range(num_beams):
            angle = i * 2 * np.pi / num_beams
            x = r * minibeam_spacing * np.cos(angle)
            y = r * minibeam_spacing * np.sin(angle)
            minibeam_centers.append([x, y])
    minibeam_centers.append([0, 0])  # central beam

elif layout_type == "radial":
    for angle in np.linspace(0, 2 * np.pi, num_minibeams_per_side**2, endpoint=False):
        x = minibeam_spacing * num_minibeams_per_side * np.cos(angle)
        y = minibeam_spacing * num_minibeams_per_side * np.sin(angle)
        minibeam_centers.append([x, y])

elif layout_type == "random":
    count = 0
    while count < num_minibeams_per_side**2:
        x = np.random.uniform(-15, 15)
        y = np.random.uniform(-15, 15)
        new = np.array([x, y])
        if all(np.linalg.norm(new - np.array(c)) > minibeam_spacing * 0.8 for c in minibeam_centers):
            minibeam_centers.append(new)
            count += 1

minibeam_centers = np.array(minibeam_centers)

# Generate clustered protons
proton_positions = []
for center in minibeam_centers:
    cluster_x = np.random.normal(center[0], minibeam_width / 2, protons_per_minibeam)
    cluster_y = np.random.normal(center[1], minibeam_width / 2, protons_per_minibeam)
    proton_positions.append(np.stack([cluster_x, cluster_y], axis=-1))

proton_positions = np.vstack(proton_positions)
proton_positions_3d = np.hstack([proton_positions, np.full((len(proton_positions), 1), initial_z)])
centers_3d = np.hstack([minibeam_centers, np.full((len(minibeam_centers), 1), initial_z)])

# Plot
scatter = go.Scatter3d(
    x=proton_positions_3d[:, 0],
    y=proton_positions_3d[:, 2],
    z=proton_positions_3d[:, 1],
    mode='markers',
    marker=dict(size=1, color='blue', opacity=0.3),
    name='Protons in Mini-Beams'
)

centers_scatter = go.Scatter3d(
    x=centers_3d[:, 0],
    y=centers_3d[:, 2],
    z=centers_3d[:, 1],
    mode='markers',
    marker=dict(size=3, color='red'),
    name='Mini-Beam Centers'
)

fig = go.Figure(data=[scatter, centers_scatter])

fig.update_layout(
    scene=dict(
        xaxis_title='X mm',
        yaxis_title='Z mm',
        zaxis_title='Y mm',
        xaxis=dict(range=[-30, 30]),
        yaxis=dict(range=[-160, 50]),
        zaxis=dict(range=[-30, 30])
    ),
    title=f"pMBRT Mini-Beam Pattern: {layout_type.capitalize()}",
    width=800,
    height=600
)

fig.show()
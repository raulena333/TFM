import numpy as np

num_minibeams_per_side = 2
minibeam_spacing = 4.0  # mm
minibeam_sigma = 0.1667  # mm (3 sigma ~ 0.5 mm cutoff)
minibeam_cutoff = 0.5    # mm
total_protons = 1000
protons_per_minibeam = total_protons // (num_minibeams_per_side ** 2)
beam_energy = 200  # MeV

grid_range = (np.arange(num_minibeams_per_side) - num_minibeams_per_side // 2) * minibeam_spacing
x_centers, y_centers = np.meshgrid(grid_range, grid_range)
x_centers = x_centers.flatten()
y_centers = y_centers.flatten()

topas_lines = []
topas_lines.append("# Auto-generated TOPAS input for 5x5 minibeam array\n")

for i, (xc, yc) in enumerate(zip(x_centers, y_centers)):
    group_name = f"Minibeam_{i}"
    topas_lines.append(f"s:Ge/{group_name}/Type = \"Group\"")
    topas_lines.append(f"s:Ge/{group_name}/Parent = \"World\"")
    topas_lines.append(f"d:Ge/{group_name}/TransX = {xc:.3f} mm")
    topas_lines.append(f"d:Ge/{group_name}/TransY = {yc:.3f} mm")
    topas_lines.append(f"d:Ge/{group_name}/TransZ = 150. mm")
    topas_lines.append("")

    topas_lines.append(f"s:So/{group_name}/Type = \"Beam\"")
    topas_lines.append(f"s:So/{group_name}/Component = \"{group_name}\"")
    topas_lines.append("s:So/{0}/BeamParticle = \"proton\"".format(group_name))
    topas_lines.append("s:So/{0}/BeamPositionDistribution = \"Gaussian\"".format(group_name))
    topas_lines.append("s:So/{0}/BeamPositionCutoffShape = \"Rectangle\"".format(group_name))
    topas_lines.append("d:So/{0}/BeamPositionSpreadX = {1:.4f} mm".format(group_name, minibeam_sigma))
    topas_lines.append("d:So/{0}/BeamPositionSpreadY = {1:.4f} mm".format(group_name, minibeam_sigma))
    topas_lines.append("d:So/{0}/BeamPositionCutoffX = {1:.1f} mm".format(group_name, minibeam_cutoff))
    topas_lines.append("d:So/{0}/BeamPositionCutoffY = {1:.1f} mm".format(group_name, minibeam_cutoff))
    topas_lines.append("s:So/{0}/BeamAngularDistribution = \"None\"".format(group_name))
    topas_lines.append("s:So/{0}/BeamEnergySpectrumType = \"Discrete\"".format(group_name))
    topas_lines.append("dv:So/{0}/BeamEnergySpectrumValues = 1 {1} MeV".format(group_name, beam_energy))
    topas_lines.append("uv:So/{0}/BeamEnergySpectrumWeights = 1 1.0".format(group_name))
    topas_lines.append("i:So/{0}/NumberOfHistoriesInRun = {1}".format(group_name, protons_per_minibeam))
    topas_lines.append("")

# Save to file
topas_input_path = "./minibeamsTopas.txt"
with open(topas_input_path, "w") as f:
    f.write("\n".join(topas_lines))

topas_input_path
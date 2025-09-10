import numpy as np
import os

# --- Configuración y Carga de Datos ---
# Define the number of protons used in the simulation for normalization purposes.
protons = 1e7

# Define voxel dimensions for the simulation volume.
voxelBig = np.array((100., 100., 150.), dtype=np.float32)
voxelShapeBins = np.array((50, 50, 300), dtype=np.int32)
voxelSize = 2 * voxelBig / voxelShapeBins
zRange = np.linspace(-voxelBig[2], voxelBig[2], voxelShapeBins[2])

# Define the paths to the data files, which are assumed to be in the correct location.
fileNameTOPAS = "../../energyDepositedTOPAS.npy"
numpyPathTrans = "./NumpyTrans/"
numpyPathNorm = "./NumpyNorm/"
fileNameSimulationTrans = f'{numpyPathTrans}energyDepositedtransformation_1e-08.npy'
fileNameSimulationNorm = f'{numpyPathNorm}energyDepositednormalization_1e-07.npy'

# Load data from three simulations into NumPy arrays.
try:
    sim1 = np.load(fileNameTOPAS) # TOPAS reference
    sim2 = np.load(fileNameSimulationTrans) # Transformation model
    sim3 = np.load(fileNameSimulationNorm) # Normalization model
except FileNotFoundError as e:
    print(f"Error: The following file was not found: {e.filename}")
    print("Please ensure the data files are in the correct location relative to the script.")
    exit()

print("--- Cuantificación de Diferencias entre Modelos ---")
print("---------------------------------------------------\n")

# --- 1. Análisis del Volumen 3D Completo ---
# Calculate the difference arrays for the two models relative to the TOPAS reference.
diff_trans = sim1 - sim2
diff_norm = sim1 - sim3

# Calculate the raw norms of the reference data for relative error calculations.
L1_topas = np.sum(np.abs(sim1))
L2_topas = np.linalg.norm(sim1)

# Calculate the raw norms (L1 and L2) of the difference arrays.
L1_trans = np.sum(np.abs(diff_trans))
L1_norm = np.sum(np.abs(diff_norm))
L2_trans = np.linalg.norm(diff_trans)
L2_norm = np.linalg.norm(diff_norm)

# Calculate the relative norms, which represent the total error as a percentage of the reference data's total value.
relative_L1_trans = L1_trans / L1_topas
relative_L1_norm = L1_norm / L1_topas
relative_L2_trans = L2_trans / L2_topas
relative_L2_norm = L2_norm / L2_topas

# Calculate L1 and L2 norms per proton for the 3D volume.
L1_trans_per_proton = L1_trans / protons
L1_norm_per_proton = L1_norm / protons
L2_trans_per_proton = L2_trans / protons
L2_norm_per_proton = L2_norm / protons

print("1.a Normas Relativas del Volumen 3D (Error en %)")
print(f"Relative L1 Transformation: {relative_L1_trans:.4f} ({relative_L1_trans*100:.2f}%)")
print(f"Relative L1 Normalization: {relative_L1_norm:.4f} ({relative_L1_norm*100:.2f}%)")
print(f"Relative L2 Transformation: {relative_L2_trans:.4f} ({relative_L2_trans*100:.2f}%)")
print(f"Relative L2 Normalization: {relative_L2_norm:.4f} ({relative_L2_norm*100:.2f}%)")
print("\n")
print("1.b Normas del Volumen 3D por Protón (MeV/protón)")
print(f"L1 Transformation per Proton: {L1_trans_per_proton:.4e}")
print(f"L1 Normalization per Proton: {L1_norm_per_proton:.4e}")
print(f"L2 Transformation per Proton: {L2_trans_per_proton:.4e}")
print(f"L2 Normalization per Proton: {L2_norm_per_proton:.4e}")
print("\n")

# Calculate the size of the array for averaging errors per voxel.
N = sim1.size
# Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).
MAE_trans = L1_trans / N
MAE_norm = L1_norm / N
RMSE_trans = L2_trans / np.sqrt(N)
RMSE_norm = L2_norm / np.sqrt(N)

print("1.c Errores Promedio por Voxel (MeV)")
print(f"MAE Transformation: {MAE_trans:.4e}")
print(f"MAE Normalization: {MAE_norm:.4e}")
print(f"RMSE Transformation: {RMSE_trans:.4e}")
print(f"RMSE Normalization: {RMSE_norm:.4e}")
print("\n")

# --- 2. Análisis del Perfil 1D (Eje Z - Profundidad) ---
# Sum over x and y axes to get the energy deposition profile along the Z-axis (depth).
sim1Z = np.sum(sim1, axis=(0, 1))
sim2Z = np.sum(sim2, axis=(0, 1))
sim3Z = np.sum(sim3, axis=(0, 1))

# Norms of the Z profiles.
diff_z_trans = sim2Z - sim1Z
diff_z_norm = sim3Z - sim1Z
L1_z_trans = np.sum(np.abs(diff_z_trans))
L2_z_trans = np.linalg.norm(diff_z_trans)
L1_z_norm = np.sum(np.abs(diff_z_norm))
L2_z_norm = np.linalg.norm(diff_z_norm)

# Calculate the relative norms for the Z profiles.
L1_topas_z = np.sum(np.abs(sim1Z))
L2_topas_z = np.linalg.norm(sim1Z)
relative_L1_z_trans = L1_z_trans / L1_topas_z
relative_L1_z_norm = L1_z_norm / L1_topas_z
relative_L2_z_trans = L2_z_trans / L2_topas_z
relative_L2_z_norm = L2_z_norm / L2_topas_z

# Calculate L1 and L2 norms per proton for the Z profile.
L1_z_trans_per_proton = L1_z_trans / protons
L1_z_norm_per_proton = L1_z_norm / protons
L2_z_trans_per_proton = L2_z_trans / protons
L2_z_norm_per_proton = L2_z_norm / protons

print("2.a Normas del Perfil 1D (Eje Z)")
print(f"Relative L1 Transformation: {relative_L1_z_trans:.4f} ({relative_L1_z_trans*100:.2f}%)")
print(f"Relative L1 Normalization: {relative_L1_z_norm:.4f} ({relative_L1_z_norm*100:.2f}%)")
print(f"Relative L2 Transformation: {relative_L2_z_trans:.4f} ({relative_L2_z_trans*100:.2f}%)")
print(f"Relative L2 Normalization: {relative_L2_z_norm:.4f} ({relative_L2_z_norm*100:.2f}%)")
print("\n")
print("2.b Normas del Perfil 1D (Eje Z) por Protón (MeV/protón)")
print(f"L1 Transformation per Proton: {L1_z_trans_per_proton:.4e}")
print(f"L1 Normalization per Proton: {L1_z_norm_per_proton:.4e}")
print(f"L2 Transformation per Proton: {L2_z_trans_per_proton:.4e}")
print(f"L2 Normalization per Proton: {L2_z_norm_per_proton:.4e}")
print("\n")

# --- 3. Análisis del Perfil 1D (Eje X en el Pico de Bragg) ---
# Find the index of the TOPAS Bragg peak (maximum energy deposition along Z-axis).
bragg_peak_index = np.argmax(sim1Z)

# Sum over y and z axes to get the energy deposition profile along the X-axis at the Bragg peak depth.
sim1X_at_bragg_peak = np.sum(sim1[:, :, bragg_peak_index], axis=1)
sim2X_at_bragg_peak = np.sum(sim2[:, :, bragg_peak_index], axis=1)
sim3X_at_bragg_peak = np.sum(sim3[:, :, bragg_peak_index], axis=1)

# Norms of the X profiles at the Bragg peak.
diff_x_trans = sim1X_at_bragg_peak - sim2X_at_bragg_peak
diff_x_norm = sim1X_at_bragg_peak - sim3X_at_bragg_peak
L1_x_trans = np.sum(np.abs(diff_x_trans))
L2_x_trans = np.linalg.norm(diff_x_trans)
L1_x_norm = np.sum(np.abs(diff_x_norm))
L2_x_norm = np.linalg.norm(diff_x_norm)

# Calculate the relative norms for the X profiles at the Bragg peak.
L1_topas_x = np.sum(np.abs(sim1X_at_bragg_peak))
L2_topas_x = np.linalg.norm(sim1X_at_bragg_peak)
relative_L1_x_trans = L1_x_trans / L1_topas_x
relative_L1_x_norm = L1_x_norm / L1_topas_x
relative_L2_x_trans = L2_x_trans / L2_topas_x
relative_L2_x_norm = L2_x_norm / L2_topas_x

# Calculate L1 and L2 norms per proton for the X profile at Bragg peak.
L1_x_trans_per_proton = L1_x_trans / protons
L1_x_norm_per_proton = L1_x_norm / protons
L2_x_trans_per_proton = L2_x_trans / protons
L2_x_norm_per_proton = L2_x_norm / protons

print("3.a Normas del Perfil 1D (Eje X en el Pico de Bragg)")
print(f"Relative L1 Transformation: {relative_L1_x_trans:.4f} ({relative_L1_x_trans*100:.2f}%)")
print(f"Relative L1 Normalization: {relative_L1_x_norm:.4f} ({relative_L1_x_norm*100:.2f}%)")
print(f"Relative L2 Transformation: {relative_L2_x_trans:.4f} ({relative_L2_x_trans*100:.2f}%)")
print(f"Relative L2 Normalization: {relative_L2_x_norm:.4f} ({relative_L2_x_norm*100:.2f}%)")
print("\n")
print("3.b Normas del Perfil 1D (Eje X en el Pico de Bragg) por Protón (MeV/protón)")
print(f"L1 Transformation per Proton: {L1_x_trans_per_proton:.4e}")
print(f"L1 Normalization per Proton: {L1_x_norm_per_proton:.4e}")
print(f"L2 Transformation per Proton: {L2_x_trans_per_proton:.4e}")
print(f"L2 Normalization per Proton: {L2_x_norm_per_proton:.4e}")
print("\n")

# --- 4. Análisis del 'Distal Fall-off' ---
# Define a threshold for fall-off (e.g., 1% of the max Bragg peak of TOPAS)
bragg_peak_max = np.max(sim1Z)
threshold_z_falloff = bragg_peak_max * 0.01

# Find the index of the TOPAS Bragg peak
bragg_peak_index = np.argmax(sim1Z)

# Find the first index after the peak where the profile falls below the threshold
falloff_topas_idx = np.where(sim1Z[bragg_peak_index:] < threshold_z_falloff)[0][0] + bragg_peak_index
falloff_trans_idx = np.where(sim2Z[bragg_peak_index:] < threshold_z_falloff)[0][0] + bragg_peak_index
falloff_norm_idx = np.where(sim3Z[bragg_peak_index:] < threshold_z_falloff)[0][0] + bragg_peak_index

# Calculate the deviation in millimeters
voxel_size_z = zRange[1] - zRange[0]
deviation_trans_mm = (falloff_trans_idx - falloff_topas_idx) * voxel_size_z
deviation_norm_mm = (falloff_norm_idx - falloff_topas_idx) * voxel_size_z

print("4. Desviación del 'Fall-off' (Distal Fall-off)")
print(f"Fall-off deviation (Transformation): {deviation_trans_mm:.2f} mm")
print(f"Fall-off deviation (Normalization): {deviation_norm_mm:.2f} mm")
print("\n")
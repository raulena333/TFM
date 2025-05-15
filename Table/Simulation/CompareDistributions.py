import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, ks_2samp

df1 = pd.read_csv('SamplingDataTrans_G4_WATER_14.9MeV.csv')
df2 = pd.read_csv('SamplingDataNorm_G4_WATER_14.9MeV.csv')
assert df1.shape == df2.shape, "Mismatch in number of samples"

plt.figure(figsize=(12, 5))

# Angles
plt.subplot(1, 2, 1)
sns.histplot(df1['angle'], bins=100, stat='density', label='Method Transform', color='blue', kde=False)
sns.histplot(df2['angle'], bins=100, stat='density', label='Method Normalize', color='orange', kde=False, alpha=0.6)
plt.xlabel("Angle")
plt.ylabel("Density")
plt.title("Angle Comparison")
plt.legend()

# Energies
plt.subplot(1, 2, 2)
sns.histplot(df1['scattered_energy'], bins=100, stat='density', label='Method Transform', color='blue', kde=False)
sns.histplot(df2['scattered_energy'], bins=100, stat='density', label='Method Normalize', color='orange', kde=False, alpha=0.6)
plt.xlabel("Scattered Energy (MeV)")
plt.ylabel("Density")
plt.title("Scattered Energy Comparison")
plt.legend()

plt.tight_layout()
plt.savefig("ComparisonOverlayHistograms.pdf")
plt.close()

# Wasserstein distance
angle_dist = wasserstein_distance(df1['angle'], df2['angle'])
energy_dist = wasserstein_distance(df1['scattered_energy'], df2['scattered_energy'])

print(f"Wasserstein Distance (Angles): {angle_dist:.4f}")
print(f"Wasserstein Distance (Energies): {energy_dist:.4f}")

# KS Test
ks_angle = ks_2samp(df1['angle'], df2['angle'])
ks_energy = ks_2samp(df1['scattered_energy'], df2['scattered_energy'])

print(f"KS Test (Angles): {ks_angle}")
print(f"KS Test (Energies): {ks_energy}")


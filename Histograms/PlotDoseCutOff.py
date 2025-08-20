import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd

# Matplotlib params
params = {
    'xtick.labelsize': 14,    
    'ytick.labelsize': 14,      
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 14
}
pylab.rcParams.update(params)

# File paths
file1 = "./ResultsWater.csv"
file2 = "./ResultsBone.csv"

# Load both CSV files
dfWater = pd.read_csv(file1)
dfBone = pd.read_csv(file2)

# Plot
plt.figure(figsize=(8, 6))

# Water
plt.scatter(dfWater['Energy (MeV)'], dfWater['Dose Cut Off (Gy)'], marker='s', s = 2, color="red", label="Water - Dose Cut Off")
plt.scatter(dfWater['Energy (MeV)'], dfWater['Dose Simulation (Gy)'], marker='s', s = 2, color="blue", label="Water - Dose Simulation")

# Bone
plt.scatter(dfBone['Energy (MeV)'], dfBone['Dose Cut Off (Gy)'], marker='s', s = 2, color="orange", label="Bone - Dose Cut Off")
plt.scatter(dfBone['Energy (MeV)'], dfBone['Dose Simulation (Gy)'], marker='s', s = 2, color="green", label="Bone- Dose Simulation")

# Labels and legend
plt.xlabel('Energy (MeV)')
plt.ylabel('Dose (Gy)')
plt.legend()
plt.tight_layout()
plt.savefig('DoseComparison.pdf', dpi=300)
plt.close()

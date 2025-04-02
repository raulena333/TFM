import h5py
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

def load_hdf5_npz(hdf5_path, npz_path, material, energy):
    # Load HDF5
    with h5py.File(hdf5_path, 'r') as file:
        energyVec_hdf5 = np.array(file[f"{material}/{energy}/EnergyVec"])
        angleVec_hdf5 = np.array(file[f"{material}/{energy}/AngleVec"])

    # Load NPZ
    npz_data = np.load(npz_path)
    energyVec_npz = npz_data[f"{material}_{energy}_energy"]
    angleVec_npz = npz_data[f"{material}_{energy}_angle"]

    return energyVec_hdf5, angleVec_hdf5, energyVec_npz, angleVec_npz

# Example
hdf5_path = "./Table/4DTableEnergy.h5"
npz_path = "./Table/4DTable.npz"
material = "G4_WATER" 
energy = 200.0 

energyVec_hdf5, angleVec_hdf5, energyVec_npz, angleVec_npz = load_hdf5_npz(hdf5_path, npz_path, material, energy)

print("HDF5 Energy Vector:", energyVec_hdf5)
print("HDF5 Angle Vector:", angleVec_hdf5)  
print("NPZ Energy Vector:", energyVec_npz)
print("NPZ Angle Vector:", angleVec_npz)
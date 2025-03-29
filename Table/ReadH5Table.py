import h5py
import numpy as np

import h5py
import numpy as np

def read_data_from_hdf5(hdf5FileName, material, initialEnergy):
    """
    Reads the energy and angle data for a specific material and initial energy
    from the HDF5 file.

    Parameters:
    - hdf5FileName (str): The path to the HDF5 file where the data is stored.
    - material (str): The material name.
    - initialEnergy (float): The initial energy of the proton beam.

    Returns:
    - energyVec (numpy.ndarray): Array of energies for the given material and energy.
    - angleVec (numpy.ndarray): Array of angles for the given material and energy.
    """
    try:
        # Open the HDF5 file
        with h5py.File(hdf5FileName, 'r') as f:
            # Check if the material exists in the file
            if material not in f:
                raise ValueError(f"Material '{material}' not found in the HDF5 file.")
            
            # Print the available energy groups for the material
            # print(f"Available energy groups for material '{material}':")
            # print(list(f[material].keys()))  # This prints all the energy values (keys)

            # Check if the initialEnergy group exists for the given material
            if str(initialEnergy) not in f[material]:
                raise ValueError(f"Energy {initialEnergy} not found for material '{material}'.")

            # Retrieve the energy and angle data for the specified material and energy
            energyVec = f[material][str(initialEnergy)]['EnergyVec'][:]
            angleVec = f[material][str(initialEnergy)]['AngleVec'][:]
        
        print(f"Successfully loaded data for material: {material} and energy: {initialEnergy}")
        return energyVec, angleVec

    except Exception as e:
        print(f"Error reading data: {e}")
        return None, None

if __name__ == "__main__":
    # Set the path to the HDF5 file and the material and energy values
    hdf5FileName = '4DTableEnergy.h5'  # Change this to the correct path if needed
    material = 'G4_WATER'
    initialEnergy = 197.2999999999999  # Replace with the energy value you're interested in

    # Read the energy and angle vectors for the specified material and energy
    energyVec, angleVec = read_data_from_hdf5(hdf5FileName, material, initialEnergy)
    
    if energyVec is not None and angleVec is not None:
        print(f"Energy Vector: {energyVec}")
        print(f"Angle Vector: {angleVec}")

import os
import re
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pylab as pylab

# Plot appearance settings
params = {
    'xtick.labelsize': 17,    
    'ytick.labelsize': 17,      
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'legend.fontsize': 15
}
pylab.rcParams.update(params)   

def modifyPositionBeam(filePath, newPositionX, newPositionY):
    patternX = r"(d:Ge/BeamPosition/TransX = )(-?\d+(\.\d+)?)( mm)"
    patternY = r"(d:Ge/BeamPosition/TransY = )(-?\d+(\.\d+)?)( mm)"

    with open(filePath, 'r') as file:
        content = file.read()
    
    # Replace the seed value in the matched line
    updatedFile = re.sub(patternX, rf"d:Ge/BeamPosition/TransX = {newPositionX} mm", content) 
    updatedFile = re.sub(patternY, rf"d:Ge/BeamPosition/TransY = {newPositionY} mm", updatedFile) 
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)
        
    print(f"Position of the beam has been modified to: X={newPositionX} mm, Y={newPositionY} mm")

    
def runTopas(filePath, dataPath):
    """
    Run TOPAS txt-script with the modified input file

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - dataPath (str): Path to the TOPAS G4 data.
    """
    try:
        result = subprocess.run(f"export TOPAS_G4_DATA_DIR={dataPath} && ~/topas/bin/topas {filePath}", 
               text=True, shell=True)
        if result == 0:
            print("Data loaded and simulation have started succesfully ")
            
    except FileNotFoundError:
        print("TOPAS executable not found. Make sure TOPAS is installed and in your PATH.")

def modifySeed(filePath, newSeed):
    """
    Modify the seed in the TOPAS input file and save it with the same name.

    Parameters:
    - filePath (str): Path to the TOPAS input file.
    - newSeed (int): New seed value to replace in the file.
    """
    patternSeed = r"i:Ts/Seed = \d+"
    
    with open(filePath, 'r') as file:
        content = file.read()
    
    # Replace the seed value in the matched line
    updatedFile = re.sub(patternSeed, rf"i:Ts/Seed = {newSeed}", content) 
    
    with open(filePath, 'w') as file:
        file.write(updatedFile)

    print(f"Updated seed to {newSeed} in {filePath}.")
    
def returnEnergyValueArray(energyFile):
    if not os.path.exists(energyFile):
        print(f"Error: File {energyFile} does not exist.")
        return None
    
    # Read the CSV file and assign column names
    data = pd.read_csv(energyFile, header=None, comment='#', names=['x', 'y', 'z', 'energy'])

    # Extract x, y, z, energy as separate numpy arrays
    x = data['x'].values
    y = data['y'].values
    z = data['z'].values
    energy = data['energy'].values
    
    return x, y, z, energy
       
if __name__ == "__main__":
    # Arguments
    dataPath = '~/G4Data/'
    voxelPhaseFile = "./TumourGeometryTsImage.txt"
    energyFile = "./EnergyAtPatientByBinsTOPAS.csv"

    numMinibeamsPerSide = 10
    minibeamSpacing = 3.0  # mm
    minibeamSigma = 0.1667  # mm (3 sigma ~ 0.5 mm cutoff)
    minibeamCutoff = 0.5    # mm
    
    voxelShapeBins = (50, 50, 300)
    grid = np.zeros(voxelShapeBins, dtype=np.float32)
    
    gridRange = (np.arange(numMinibeamsPerSide) - numMinibeamsPerSide // 2) * minibeamSpacing
    xCenters, yCenters = np.meshgrid(gridRange, gridRange)
    xCenters = xCenters.flatten()
    yCenters = yCenters.flatten()
    
    for i, (xc, yc) in enumerate(zip(xCenters, yCenters)):
        modifyPositionBeam(voxelPhaseFile, xc, yc)
        
        # Create new random seed for each minibeam
        newSeed = np.random.randint(0, 10000000)
        modifySeed(voxelPhaseFile, newSeed) 
        
        # Run TOPAS
        runTopas(voxelPhaseFile, dataPath)
        
        # Deposit energy
        x, y, z, energies = returnEnergyValueArray(energyFile)
        
        # Only keep non-zero energy deposits
        mask = energies > 0
        x = x[mask]
        y = y[mask]
        z = z[mask]
        energies = energies[mask]
        z = voxelShapeBins[2] - 1 - z
        
        for xi, yi, zi, ei in zip(x, y, z, energies):
            grid[xi, yi, zi] += ei
        
    np.save('projectionXZTOPAS.npy', grid)
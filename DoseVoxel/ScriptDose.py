import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pylab as pylab

params = {
    'xtick.labelsize': 15,    
    'ytick.labelsize': 15,      
    'axes.titlesize' : 15,
    'axes.labelsize' : 15,
    'legend.fontsize': 15
}
pylab.rcParams.update(params)  # Apply changes

def returnDoseValue(fileName):
    """
    Open the voxel file and return the value of the dose in that voxel.

    Parameters:
    - fileName (str): Path to the TOPAS output file.

    Returns:
    - float: The dose value
    """
    if not os.path.exists(fileName):
        print(f"Error: File {fileName} does not exist.")
        return None

    # Load data, skipping first 5 rows
    data = np.loadtxt(fileName, skiprows=5)

    # Return the first row's value (or modify as needed)
    return data


if __name__ == "__main__":
    filesName = "OutputVoxel"
    termination = ".csv"
    
    # Grid dimensions (3x3)
    gridSize = 3  
    doseValues = np.zeros((gridSize, gridSize))  # Initialize a 3x3 array

    # Define the custom mapping for order
    mapping = [
        (2, 0), (1, 0), (0, 0),  
        (2, 1), (1, 1), (0, 1),  
        (2, 2), (1, 2), (0, 2)   
    ]

    for i in range(1, gridSize * gridSize + 1):
        fileName = f"{filesName}{i}{termination}"
        doseValue = returnDoseValue(fileName)

        if doseValue is not None:
            x, y = mapping[i - 1]  # Get(row, col) position
            doseValues[x, y] = doseValue

    # Plot
    extent = [-1.5, 1.5, -1.5, 1.5]  # Define the physical size of the grid
    fig = plt.figure(figsize=(8, 6.4))
    norm = mcolors.LogNorm(vmin=np.min(doseValues) + 1e-5, vmax=np.max(doseValues))  
    plt.imshow(doseValues, cmap='inferno', interpolation='nearest', origin='upper', extent=extent)  
    plt.colorbar(label="Dose Value")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")

    plt.savefig("VoxelDoseComparation.pdf")
    # plt.show()

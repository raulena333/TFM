import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from pymedphys import gamma as pymedphys_gamma
import time

# Matplotlib params
params = {
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'axes.titlesize': 17,
    'axes.labelsize': 17,
    'legend.fontsize': 12
}
pylab.rcParams.update(params)  # Apply changes

def readandProcessCSVFiles(topasPath, varTransPath, normPath):
    '''
    Read the CSV files from the given paths and return the dataframes.
    '''
    # Read the CSV files# Read the CSV, skipping lines starting with '#'
    dataTOPAS = pd.read_csv(topasPath, comment='#', sep=',', header=None)
    dataTrans = pd.read_csv(varTransPath, comment='#', sep='\s+', header=None)
    dataNormalized = pd.read_csv(normPath, comment='#', sep='\s+', header=None)

    xTOPAS = dataTOPAS[0].astype(int).values
    yTOPAS = dataTOPAS[1].astype(int).values
    zTOPAS = dataTOPAS[2].astype(int).values
    energyDepositTOPAS = dataTOPAS[3].values

    # Create a npdarray to hold the data
    energyTOPAS = np.zeros((len(np.unique(xTOPAS)), len(np.unique(yTOPAS)), len(np.unique(zTOPAS))))
    
    for i in range(len(xTOPAS)):
        energyTOPAS[xTOPAS[i], yTOPAS[i], zTOPAS[i]] = energyDepositTOPAS[i]
        
    xTrans = dataTrans[0].astype(int).values
    yTrans = dataTrans[1].astype(int).values
    zTrans = dataTrans[2].astype(int).values
    energyDepositTrans = dataTrans[3].values
        
    # Create a npdarray to hold the data
    energyTrans = np.zeros((len(np.unique(xTrans)), len(np.unique(yTrans)), len(np.unique(zTrans))))
    
    for i in range(len(xTrans)):
        energyTrans[xTrans[i], yTrans[i], zTrans[i]] = energyDepositTrans[i]
        
    xNorm = dataNormalized[0].astype(int).values
    yNorm = dataNormalized[1].astype(int).values
    zNorm = dataNormalized[2].astype(int).values
    energyDepositNorm = dataNormalized[3].values
        
    # Create a npdarray to hold the data
    energyNormalized = np.zeros((len(np.unique(xNorm)), len(np.unique(yNorm)), len(np.unique(zNorm))))
    
    for i in range(len(xNorm)):
        energyNormalized[xNorm[i], yNorm[i], zNorm[i]] = energyDepositNorm[i]
    
    return energyTOPAS, energyTrans, energyNormalized

def calculateGammaIndex(
    energyTOPAS: np.ndarray,
    energyTrans: np.ndarray,
    axes: list[np.ndarray], 
    dosePercentThreshold: float, 
    dtaMmThreshold: float, 
    lowerDoseCutoffPercent: float = 10.0,
    globalDoseReference: float | None = None,
    interpFraction: int = 10,
    maxGamma: float = 2.0,
    randomSubset: int | None = None,
    quiet : bool = False
    ) -> tuple[np.ndarray, float]:
    """
    Calculate the Gamma Index from two dataframes of dose distributions.

    Parameters
    ----------
    energyTOPAS: np.ndarray
        A 3D numpy array containing the dose distribution from TOPAS.
    energyTrans: np.ndarray
        A 3D numpy array containing the dose distribution from Trans.
    axes: list[np.ndarray]
        A list of numpy arrays containing the x, y, and z axes for each dimension.
    dosePercentThreshold: float
        The percent dose threshold for the Gamma Index calculation.
    dtaMmThreshold: float
        The distance-to-agreement in mm for the Gamma Index calculation.
    lowerDoseCutoffPercent: float, optional
        Exclude points in the reference dose below this
                percentage of global_dose_reference from gamma
                calculation. Defaults to 10%.
    globalDoseReference: float | None, optional
        The dose value (e.g., max dose, prescription dose)
                used for calculating the dose difference threshold.
                If None, uses the maximum dose in dose_ref.
    interpFraction: int, optional
        The interpolation fraction for the Gamma Index calculation. The fraction 
                of voxel size to sample doses for interpolation. 0 means no 
                interpolation. Defaults to 10.
    maxGamma: float, optional
        The maximum Gamma Index value to consider.
    randomSubset: int | None, optional
        Calculate gamma for only a random subset of points.
                    Useful for quick checks on large datasets. Defaults to None.
    quiet : bool, optional
        If True, suppresses the output of the Gamma Index calculation.

    Returns
    -------
    tuple[np.ndarray, float]
        - gammaMap: 3D numpy array of the calculated gamma index for each voxel
                     in the reference grid. Returns None if pymedphys is not installed.
                     Voxels below the lower_dose_cutoff will have NaN values.
        - passRate: The percentage of valid voxels (not NaN) where gamma <= 1.0.
                     Returns None if pymedphys is not installed or calculation fails.
    """
    
    start_time = time.time()
    
    # Determine the reference dose for threshold calculation if not provided
    calcGlobalDoseReference = globalDoseReference
    if calcGlobalDoseReference is None:
        calcGlobalDoseReference = np.max(energyTOPAS)
        if not quiet:
            print(f"Using max reference dose as global reference: {calcGlobalDoseReference:.2f}")

    if calcGlobalDoseReference == 0:
         raise ValueError("Global dose reference cannot be zero.")

    doseDifferenceThresholdValue = (dosePercentThreshold/ 100.0) * calcGlobalDoseReference
    lowerDoseCutoffValue = (lowerDoseCutoffPercent / 100.0) * calcGlobalDoseReference

    if not quiet:
        print(f"Calculating Gamma using pymedphys...")
        print(f"  Criteria: DTA={dtaMmThreshold} mm, DD={dosePercentThreshold}% ({doseDifferenceThresholdValue:.2f})")
        print(f"  Lower dose cutoff: {lowerDoseCutoffPercent}% ({lowerDoseCutoffValue:.2f})")
        print(f"  Interpolation fraction: {interpFraction}")
        print(f"  Max Gamma: {maxGamma}")
        if randomSubset:
            print(f"  Using random subset: {randomSubset} points")
                
    try:
        # Call the pymedphys gamma function
        gammaMap = pymedphys_gamma(
            axes_reference=axes,
            dose_reference=energyTOPAS,
            axes_evaluation=axes,
            dose_evaluation=energyTrans,
            dose_percent_threshold=dosePercentThreshold,
            distance_mm_threshold=dtaMmThreshold,
            lower_percent_dose_cutoff=lowerDoseCutoffPercent,
            global_normalisation=globalDoseReference, 
            interp_fraction=interpFraction,
            max_gamma=maxGamma,
            random_subset=randomSubset,
            quiet=quiet # Pass quiet status through
        )

        end_time = time.time()
        if not quiet:
            print(f"pymedphys calculation finished in {end_time - start_time:.4f} seconds.")

        # --- Calculate Pass Rate ---
        # Create a mask for valid voxels in the reference dose grid
        # Note: pymedphys gamma output aligns with the reference grid
        validMask = (energyTOPAS >= lowerDoseCutoffValue) & (~np.isnan(gammaMap))

        # Count total valid voxels and passing voxels
        validVoxelsEvaluated = np.sum(validMask)
        passingVoxels = np.sum(gammaMap[validMask] <= 1.0)

        if validVoxelsEvaluated > 0:
            passRate = (passingVoxels / validVoxelsEvaluated) * 100.0
        else:
            passRate = 0.0 # Or np.nan

        if not quiet:
            print(f"Gamma analysis summary (pymedphys):")
            print(f"  Voxels evaluated (above cutoff): {validVoxelsEvaluated}")
            print(f"  Voxels passing (Gamma <= 1.0): {passingVoxels}")
            print(f"  Pass Rate: {passRate:.2f}%")

        return gammaMap, passRate

    except Exception as e:
        print(f"An error occurred during pymedphys gamma calculation: {e}")
        return None, None
    
# Main simulation function
if __name__ == "__main__":
    # Define the paths to the CSV files
    varTransPath ='../CSVTrans/EnergyAtBoxByBinsMySimulation_transformation.csv'
    topasPath = '../../../EnergyAtPatientByBinsTOPASHetero.csv'
    normPath = '../CSVNorm/EnergyAtBoxByBinsMySimulation_normalization.csv'
    
    energyTOPAS, energyTrans, energyNorm = readandProcessCSVFiles(topasPath, varTransPath, normPath)
    energyTrans = np.flip(energyTrans, axis=2)
    energyNorm = np.flip(energyNorm, axis=2)
    
    voxelSpacing = (4, 4, 1) # voxel sizes 4 mm for X and Y and 1 mm for Z 
    sizeVolume = (200, 200, 300) # 200 mm for X and Y and 300 mm for Z
    voxelBins = (50, 50, 300)
    
    # Coordinate axes centered at (0, 0, 0)
    xCoords = np.linspace(-100 + voxelSpacing[0] / 2, 100 - voxelSpacing[0] / 2, voxelBins[0])
    yCoords = np.linspace(-100 + voxelSpacing[1] / 2, 100 - voxelSpacing[1] / 2, voxelBins[1])
    zCoords = np.linspace(-150 + voxelSpacing[2] / 2, 150 - voxelSpacing[2] / 2, voxelBins[2])
    
    # Combine axes for pymedphys
    axes = (xCoords, yCoords, zCoords)
    
    # --- Gamma Calculation Parameters ---
    ddPercent = 2.0  # 2% Dose Difference
    dtaMm = 1.0      # 1mm Distance-to-Agreement
    lowerCutoff = 0.001 # 10% lower dose cutoff

    # --- Run pymedphys Gamma ---
    gammaMapPymed_trans, passRatePymed_trans = calculateGammaIndex(
        energyTOPAS=energyTOPAS,
        energyTrans=energyTrans,
        axes = axes,
        dosePercentThreshold = ddPercent,
        dtaMmThreshold = dtaMm,
        lowerDoseCutoffPercent = lowerCutoff,
        interpFraction = 10,
        maxGamma = 2.0,
        randomSubset = None,
        quiet = False
    )
                       
    gammaMapPymed_norm, passRatePymed_norm = calculateGammaIndex(
        energyTOPAS=energyTOPAS,
        energyTrans=energyNorm,
        axes = axes,
        dosePercentThreshold = ddPercent,
        dtaMmThreshold = dtaMm,
        lowerDoseCutoffPercent = lowerCutoff,
        interpFraction = 10,
        maxGamma = 2.0,
        randomSubset = None,
        quiet = False
    )

# Flip gamma maps along Z axis after calculation for correct left-to-right plotting
gammaMapPymed_trans = np.flip(gammaMapPymed_trans, axis=2)
gammaMapPymed_norm = np.flip(gammaMapPymed_norm, axis=2)

# Plot middle slice for both gamma maps side by side with pass rates
try:
    sliceIdx = voxelBins[0] // 2  # middle slice in X
    x_val = axes[0][sliceIdx]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # --- Transformation Gamma Map ---
    im0 = axs[0].imshow(
        gammaMapPymed_trans[sliceIdx, :, :], 
        cmap='coolwarm', vmin=0, vmax=2,
        extent=[axes[2][0], axes[2][-1], axes[1][0], axes[1][-1]],  # left-to-right
        origin='lower'  # ensures lower Y is at bottom
    )
    axs[0].set_title(f'Transformation Gamma\nPass Rate: {passRatePymed_trans:.2f}% (X={x_val:.1f} mm)')
    axs[0].set_xlabel('Z (mm)')
    axs[0].set_ylabel('Y (mm)')
    fig.colorbar(im0, ax=axs[0], label='Gamma Index')

    # --- Normalization Gamma Map ---
    im1 = axs[1].imshow(
        gammaMapPymed_norm[sliceIdx, :, :], 
        cmap='coolwarm', vmin=0, vmax=2,
        extent=[axes[2][0], axes[2][-1], axes[1][0], axes[1][-1]],  # left-to-right
        origin='lower'
    )
    axs[1].set_title(f'Normalization Gamma\nPass Rate: {passRatePymed_norm:.2f}% (X={x_val:.1f} mm)')
    axs[1].set_xlabel('Z (mm)')
    axs[1].set_ylabel('Y (mm)')
    fig.colorbar(im1, ax=axs[1], label='Gamma Index')

    # plt.suptitle(f'Gamma Analysis Comparison ({ddPercent}% / {dtaMm} mm)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'GammaAnalysis_Comparison_{ddPercent}_{dtaMm}.pdf', bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"An error occurred during plotting: {e}")

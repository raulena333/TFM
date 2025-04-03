import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import argparse

def apply_thresholds(meanEnergies, stdEnergies, meanAngles, stdAngles, thresholds, mode='apply'):
    """Applies thresholds to all data arrays."""
    if mode == 'apply':
        mask = (meanEnergies >= thresholds['meanEnergy'][1]) & (meanEnergies <= thresholds['meanEnergy'][0]) & \
               (stdEnergies >= thresholds['stdEnergy'][1]) & (stdEnergies <= thresholds['stdEnergy'][0]) & \
               (meanAngles >= thresholds['meanAngle'][1]) & (meanAngles <= thresholds['meanAngle'][0]) & \
               (stdAngles >= thresholds['stdAngle'][1]) & (stdAngles <= thresholds['stdAngle'][0])
    elif mode == 'no-threshold':
        mask = np.ones_like(meanEnergies, dtype=bool)  # No filtering if no-threshold mode
    
    # Apply the mask to all arrays
    filtered_meanEnergies = meanEnergies[mask]
    filtered_stdEnergies = stdEnergies[mask]
    filtered_meanAngles = meanAngles[mask]
    filtered_stdAngles = stdAngles[mask]
    
    return filtered_meanEnergies, filtered_stdEnergies, filtered_meanAngles, filtered_stdAngles


def plot_histograms(meanEnergies, stdEnergies, meanAngles, stdAngles):
    """Generates and saves histograms."""
    numberOfBinsEnergies = 100
    numberOfBinsAngles = 100
    
    # Plot histograms of energy and angle
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 6))

    sns.histplot(meanEnergies, bins=numberOfBinsEnergies, edgecolor="black", color='orange', kde=False, ax=axs1[0])
    axs1[0].set_xlabel(r'$E[E_f]$ (MeV)')
    axs1[0].set_title('Final mean Energy distribution')
    axs1[0].set_yscale('log')

    sns.histplot(stdEnergies, bins=numberOfBinsEnergies, edgecolor="black", color='red', kde=False, ax=axs1[1])
    axs1[1].set_xlabel(r'$\sigma_{E_f}$ (MeV)')
    axs1[1].set_title('Final Std Energy distribution')
    axs1[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('HistogramStdMeanEnergy.pdf')
    plt.close(fig1)

    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 6))

    sns.histplot(meanAngles, bins=numberOfBinsAngles, edgecolor="black", color='orange', kde=False, ax=axs2[0])
    axs2[0].set_xlabel(r'$E[\theta]$ (deg)')
    axs2[0].set_title('Final Mean Angle distribution')
    axs2[0].set_yscale('log')

    sns.histplot(stdAngles, bins=numberOfBinsAngles, edgecolor="black", color='red', kde=False, ax=axs2[1])
    axs2[1].set_xlabel(r'$\sigma_\theta$ (deg)')
    axs2[1].set_title('Final Std Angle distribution')
    axs2[1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('HistogramStdMeanAngle.pdf')
    plt.close(fig2)

def plot_3d_distributions(meanEnergies, stdEnergies, meanAngles, stdAngles):
    """Generates and saves 3D probability distributions."""
    jointNumberOfBins = 100
    hist1, xMean, yMean = np.histogram2d(meanAngles, meanEnergies, bins=jointNumberOfBins)
    hist2, xStd, yStd = np.histogram2d(stdAngles, stdEnergies, bins=jointNumberOfBins)
    
    probabilitiesMean = hist1 / np.sum(hist1)
    probabilitiesStd = hist2 / np.sum(hist2)

    # Create grid for plotting
    Xmean, Ymean = np.meshgrid(xMean[:-1], yMean[:-1])
    Zmean = probabilitiesMean.T
    Xstd, Ystd = np.meshgrid(xStd[:-1], yStd[:-1])
    Zstd = probabilitiesStd.T

    # 3D Plot for Mean Energy and Angle Distribution
    fig3 = plt.figure(figsize=(10, 7))
    ax1 = fig3.add_subplot(111, projection='3d')
    surf = ax1.plot_surface(Xmean, Ymean, Zmean, cmap='coolwarm', edgecolor='k', alpha=0.9)
    ax1.set_xlabel(r'$E[\theta]$ (deg)')
    ax1.set_ylabel(r'$E[E_f]$ (MeV)')
    cbar = plt.colorbar(surf, ax=ax1, shrink=0.5, pad=0.05, aspect=4)
    cbar.set_label('Probability')
    plt.tight_layout()
    plt.savefig('ProbabilityDistributionMean.pdf')
    plt.close(fig3)

    # 3D Plot for Std Energy and Angle Distribution
    fig4 = plt.figure(figsize=(10, 7))
    ax2 = fig4.add_subplot(111, projection='3d')
    surf = ax2.plot_surface(Xstd, Ystd, Zstd, cmap='coolwarm', edgecolor='k', alpha=0.9)
    ax2.set_xlabel(r'$\sigma_\theta$ (deg)')
    ax2.set_ylabel(r'$\sigma_{E_f}$ (MeV)')
    cbar = plt.colorbar(surf, ax=ax2, shrink=0.5, pad=0.05, aspect=4)
    cbar.set_label('Probability')
    plt.tight_layout()
    plt.savefig('ProbabilityDistributionStd.pdf')
    plt.close(fig4)

def main(args):
    # Load HDF5 file
    hdf5_path = "SimulationMeanStd.h5"
    nProtons = 1000  # Number of protons to analyze
    
    with h5py.File(hdf5_path, 'r') as f:
        # Get the group for the specified number of protons
        numberOfProtons = f"{nProtons} nProtons"
        if numberOfProtons not in f:
            raise ValueError(f"Group {numberOfProtons} not found in HDF5 file.")
        
        # Extract data from the group
        simulations = f[numberOfProtons]

        meanEnergies, stdEnergies, meanAngles, stdAngles = [], [], [], []

        for simKey in simulations.keys():
            simGroup = simulations[simKey]
            
            # Append data ensuring consistent shape
            meanEnergy = np.ravel(simGroup['meanEnergy'][()])
            stdEnergy = np.ravel(simGroup['stdEnergy'][()])
            meanAngle = np.ravel(simGroup['meanAngle'][()])
            stdAngle = np.ravel(simGroup['stdAngle'][()])
            
            # Append to lists
            meanEnergies.extend(meanEnergy)
            stdEnergies.extend(stdEnergy)
            meanAngles.extend(meanAngle)
            stdAngles.extend(stdAngle)

        # Convert lists to numpy arrays
        meanEnergies = np.array(meanEnergies)
        stdEnergies = np.array(stdEnergies)
        meanAngles = np.array(meanAngles)
        stdAngles = np.array(stdAngles)

        print(f"Data Loaded: {len(meanEnergies)} mean energies, {len(stdEnergies)} std energies, "
              f"{len(meanAngles)} mean angles, {len(stdAngles)} std angles.")
        
        # Calculate percentage of discarded angles
        firstData = len(meanEnergies)
        
        # Apply thresholds if needed
        if args.mode == 'threshold':
            thresholds = {
                'meanEnergy': [99.65, 99.625],
                'stdEnergy': [0.1, 0],
                'meanAngle': [0.19, 0.14],
                'stdAngle': [1, 0]
            }
            # Apply thresholds and filter data
            meanEnergies, stdEnergies, meanAngles, stdAngles = apply_thresholds(
                meanEnergies, stdEnergies, meanAngles, stdAngles, thresholds, mode='apply'
            )

            print(f"Filtered Data: {len(meanEnergies)} mean energies, {len(stdEnergies)} std energies, "
                  f"{len(meanAngles)} mean angles, {len(stdAngles)} std angles.")
            
            laterData = len(meanEnergies) 
            discardedData = firstData - laterData
            print(f"Discarded Data: {discardedData}. Corresponds to {discardedData / firstData * 100:.4f}% of the original data.")
            
        # Plot the data
        plot_histograms(meanEnergies, stdEnergies, meanAngles, stdAngles)
        plot_3d_distributions(meanEnergies, stdEnergies, meanAngles, stdAngles)

if __name__ == '__main__':
    # Set up argparse
    parser = argparse.ArgumentParser(description="Analyze and visualize simulation data.")
    parser.add_argument('--mode', choices=['threshold', 'no'], default='threshold',
                        help="Mode for running the script. 'threshold' applies filters, 'no-threshold' skips them.")

    args = parser.parse_args()
    
    # Run the main function with the provided arguments
    main(args)

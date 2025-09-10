import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
import argparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker

# Matplotlib params
params = {
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'legend.fontsize': 12
}
pylab.rcParams.update(params)  # Apply changes

def createPhysicalSpace(bigVoxel, voxelShapeBins):
    """
    Creates a range of coordinates from -bigVoxel to +bigVoxel with specified bins.
    """
    xRange = np.linspace(-bigVoxel[0], bigVoxel[0], voxelShapeBins[0])
    yRange = np.linspace(-bigVoxel[1], bigVoxel[1], voxelShapeBins[1])
    zRange = np.linspace(-bigVoxel[2], bigVoxel[2], voxelShapeBins[2])
    return xRange, yRange, zRange

def get_voxel_index_z(z, z_min=-150.0, z_max=150.0, num_bins=300):
    """
    Converts a Z-coordinate (in mm) to its corresponding voxel index.
    """
    voxel_size = (z_max - z_min) / num_bins
    index = int((z - z_min) // voxel_size)
    return index

def getLateralProfile(data, z_index):
    """
    Extracts a 2D lateral profile (XY plane) from the 3D data at a given Z index.
    """
    return data[:, :, z_index]

def load_simulations(mode, numpy_path, thresholds):
    """
    Loads the main simulation files and the thresholded files based on the chosen mode.
    """
    # Load the "noisy" data file (original without threshold)
    noisy_file_path = f'{numpy_path}energyDeposited{mode}.npy'
    noisy_data = np.load(noisy_file_path)
    
    extreme_noisy_file_path = f'{numpy_path}energyDeposited{mode}_extreme.npy'
    extreme_noisy_data = np.load(extreme_noisy_file_path)
    
    # Load the thresholded files
    threshold_data = []
    for t in thresholds:
        threshold_str = f'{t:1.0e}'
        file_path = f'{numpy_path}energyDeposited{mode}_{threshold_str}.npy'
        # Load the .npy file directly, as it is not a compressed archive.
        data = np.load(file_path)
        threshold_data.append(data)

    # Combine all data into a list for easy iteration
    all_data = [noisy_data, extreme_noisy_data] + threshold_data
    labels = [f'Noisy', f'Very Noisy', f'Denoised {thresholds[0]:.0e}',
            f'Denoised {thresholds[1]:.0e}', f'Denoised {thresholds[2]:.0e}']

    return all_data, labels

def load_mean_energy_grids(mode, numpy_path, thresholds):
    """
    Loads the mean energy grid files.
    """
    # Load the "noisy" mean energy grid
    noisy_file_path = f'{numpy_path}meanEnergyGrid{mode}.npy'
    noisy_grid = np.load(noisy_file_path)
    
    extreme_noisy_file_path = f'{numpy_path}meanEnergyGrid{mode}_extreme.npy'
    extreme_noisy_grid = np.load(extreme_noisy_file_path)

    # Load the thresholded mean energy grids
    threshold_grids = []
    for t in thresholds:
        threshold_str = f'{t:1.0e}'
        file_path = f'{numpy_path}meanEnergyGrid{mode}_{threshold_str}.npy'
        # Load the .npy file directly, as it is not a compressed archive.
        data = np.load(file_path)
        threshold_grids.append(data)

    all_grids = [noisy_grid, extreme_noisy_grid] + threshold_grids
    return all_grids


if __name__ == "__main__":
    # --- Argparse Setup ---
    parser = argparse.ArgumentParser(description="Plot and compare simulation data against TOPAS reference.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--normalization", action="store_true", help="Plot normalization data.")
    group.add_argument("--transformation", action="store_true", help="Plot transformation data.")
    args = parser.parse_args()

    mode = "normalization" if args.normalization else "transformation"
    material = 'Homo'

    # --- File Paths ---
    fileNameTOPAS = "../../energyDepositedTOPAS.npy"
    numpyPathTrans = f"./NumpyTrans{material}/"
    numpyPathNorm = f"./NumpyNorm{material}/"
    thresholds = [1e-5, 5e-07, 1e-08]
    
    # Select the correct path based on the mode
    numpyPath = numpyPathNorm if mode == "normalization" else numpyPathTrans
    
    savePath = f"./PlotsBetter_{mode}/"
    os.makedirs(savePath, exist_ok=True)

    # --- Load Data ---
    voxelBig = np.array((100., 100., 150.), dtype=np.float32)
    voxelShapeBins = np.array((50, 50, 300), dtype=np.int32)
    xRange, yRange, zRange = createPhysicalSpace(voxelBig, voxelShapeBins)

    # Load the data for all simulations to be plotted
    data_topas = np.load(fileNameTOPAS)
    all_sim_data, labels = load_simulations(mode, numpyPath, thresholds)
    all_mean_grids = load_mean_energy_grids(mode, numpyPath, thresholds)
    
    print(f"Loaded TOPAS data with shape {data_topas.shape}")
    for i, data in enumerate(all_sim_data):
        print(f"Loaded {labels[i]} data with shape {data.shape}")

    # --- Plotting Functions ---
    def plot_z_profile(data_topas_z, all_sims_z, sim_labels):
        """
        Plots energy deposition vs z-depth with an inset for differences.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the main data on the primary axis
        ax.plot(zRange, data_topas_z, label='TOPAS', color='red', linestyle='-', linewidth=1)
        colors = ['blue', 'black', 'green', 'purple', 'orange']
        linestyles = ['--', '--', '--', '--', '--']

        for i, sim_z in enumerate(all_sims_z):
            ax.plot(zRange, sim_z, label=sim_labels[i], color=colors[i], linestyle=linestyles[i], linewidth=1)

        ax.set_xlabel('Z (mm)')
        ax.set_ylabel('Energy deposited (MeV)')
        ax.legend(loc='best', shadow=True, fancybox=True, framealpha=0.9)
        ax.grid(True)
        
        # Get the exponent of the main plot's y-axis scale
        y_max_main = ax.get_ylim()[1]
        main_exp = int(np.floor(np.log10(y_max_main)))

        # Add the inset axis
        ax_inset = inset_axes(ax, width="110%", height="110%", bbox_to_anchor=(0.18, 0.6, 0.3, 0.3), bbox_transform=ax.transAxes, borderpad=0)

        # Plot the differences on the inset axis
        for i, sim_z in enumerate(all_sims_z):
            diff = data_topas_z - sim_z
            ax_inset.plot(zRange, diff, label=f'TOPAS - {sim_labels[i]}', color=colors[i], linestyle='-', linewidth=0.9)

        # Configure a ScalarFormatter to use the same scientific exponent as the main plot
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((main_exp, main_exp))
        
        # Apply the formatter to the inset's y-axis
        ax_inset.yaxis.set_major_formatter(formatter)

        # Set the labels for the inset
        ax_inset.set_xlabel('Z (mm)', fontsize=13)
        ax_inset.set_ylabel('Difference (MeV)', fontsize=13)
        ax_inset.tick_params(labelsize=10)
        ax_inset.grid(True)
        
        # ax_inset.set_title('Differences', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{savePath}EnergyDepositionZ_{mode}.pdf')
        plt.close()


    def plot_lateral_profiles(data_topas, all_sim_data, sim_labels):
        """Plots X and Y lateral profiles at various z-slices."""
        braggIndexTOPAS = np.argmax(np.sum(data_topas, axis=(0, 1)))
        zSlices = {
            'Entrance(z~--150)': get_voxel_index_z(-150),
            'Zone1(z~-100)': get_voxel_index_z(-100),
            'Zone2(z~-50)': get_voxel_index_z(-50),
            'Zone3(z~0)': get_voxel_index_z(0),
            'Zone4(z~50)': get_voxel_index_z(50),
            'BraggPeak': braggIndexTOPAS
        }

        for title, zIdx in zSlices.items():
            # X-profile
            fig, ax = plt.subplots(figsize=(8, 6))
            profiles_x = [np.sum(getLateralProfile(data, zIdx), axis=0) for data in [data_topas] + all_sim_data]
            
            ax.plot(xRange, profiles_x[0], label='TOPAS', color='red', linestyle='-', linewidth=1)
            colors = ['blue', 'black', 'green', 'purple', 'orange']
            linestyles = ['--', '--', '--', '--', '--']
            for i, profile in enumerate(profiles_x[1:]):
                ax.plot(xRange, profile, label=sim_labels[i], color=colors[i], linestyle=linestyles[i], linewidth=1)
            
            ax.set_xlim(-voxelBig[0] / 2 - 5, voxelBig[0] / 2 + 5)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Energy (MeV)')
            ax.grid(True)
            ax.legend()
            
            ax_inset = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.12, 0.6, 0.3, 0.3), bbox_transform=ax.transAxes, borderpad=0)
            for i, profile in enumerate(profiles_x[1:]):
                diff = profiles_x[0] - profile
                ax_inset.plot(xRange, diff, label=f'TOPAS - {sim_labels[i]}', color=colors[i], linestyle='-')
                
            # New code for inset formatting
            y_max_main_x = ax.get_ylim()[1]
            main_exp_x = int(np.floor(np.log10(y_max_main_x)))
            formatter_x = ticker.ScalarFormatter(useMathText=True)
            formatter_x.set_scientific(True)
            formatter_x.set_powerlimits((main_exp_x, main_exp_x))
            ax_inset.yaxis.set_major_formatter(formatter_x)
        
            ax_inset.set_xlim(-voxelBig[0] / 2, voxelBig[0] / 2)
            ax_inset.set_xlabel('X (mm)', fontsize=12)
            #ax_inset.set_ylabel('Difference (MeV)', fontsize=12)
            #ax_inset.set_title('Difference (MeV)', fontsize=10)
            ax_inset.tick_params(labelsize=11)
            ax_inset.grid(True)
            plt.tight_layout()
            plt.savefig(f'{savePath}ProfileX_{title}_{mode}.pdf')
            plt.close()

            # Y-profile
            fig, ax = plt.subplots(figsize=(8, 6))
            profiles_y = [np.sum(getLateralProfile(data, zIdx), axis=1) for data in [data_topas] + all_sim_data]
            
            ax.plot(yRange, profiles_y[0], label='TOPAS', color='red', linestyle='-', linewidth=1)
            for i, profile in enumerate(profiles_y[1:]):
                ax.plot(yRange, profile, label=sim_labels[i], color=colors[i], linestyle=linestyles[i], linewidth=1)

            ax.set_xlim(-voxelBig[1] / 2, voxelBig[1] / 2)
            ax.set_xlabel('Y (mm)')
            ax.set_ylabel('Energy (MeV)')
            ax.grid(True)
            ax.legend()

            ax_inset = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.15, 0.6, 0.3, 0.3), bbox_transform=ax.transAxes, borderpad=0)
            for i, profile in enumerate(profiles_y[1:]):
                diff = profiles_y[0] - profile
                ax_inset.plot(yRange, diff, label=f'TOPAS - {sim_labels[i]}', color=colors[i], linestyle='-')
                
            # New code for inset formatting
            y_max_main_y = ax.get_ylim()[1]
            main_exp_y = int(np.floor(np.log10(y_max_main_y)))
            formatter_y = ticker.ScalarFormatter(useMathText=True)
            formatter_y.set_scientific(True)
            formatter_y.set_powerlimits((main_exp_y, main_exp_y))
            ax_inset.yaxis.set_major_formatter(formatter_y)
        
            ax_inset.set_xlim(-voxelBig[1] / 2, voxelBig[1] / 2)
            ax_inset.set_xlabel('Y (mm)', fontsize=12)
            # ax_inset.set_ylabel('Difference (MeV)', fontsize=12)
            # ax_inset.set_title('Difference (MeV)', fontsize=10)
            ax_inset.tick_params(labelsize=11)
            ax_inset.grid(True)
            plt.tight_layout()
            plt.savefig(f'{savePath}ProfileY_{title}_{mode}.pdf')
            plt.close()

    def plot_mean_energy_z_profile(grid_topas, all_sim_grids, sim_labels, xIndex, yIndex):
        """Plots mean energy profile vs z-depth with an inset for differences."""
        profileZMeanTOPAS = grid_topas[xIndex, yIndex, :]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(zRange, profileZMeanTOPAS, label='TOPAS', color='red', linestyle='-', linewidth=1)
        colors = ['blue', 'black', 'green', 'purple', 'orange']
        linestyles = ['--', '--', '--', '--', '--']
        
        for i, grid in enumerate(all_sim_grids):
            profile = grid[xIndex, yIndex, :]
            ax.plot(zRange, profile, label=sim_labels[i], color=colors[i], linestyle=linestyles[i], linewidth=1)

        ax.set_xlabel('Z (mm)')
        ax.set_ylabel('Mean Energy (MeV)')
        ax.legend(loc='best', shadow=True, fancybox=True, framealpha=0.9)
        ax.grid(True)

        ax_inset = inset_axes(ax, width="110%", height="110%", bbox_to_anchor=(0.15, 0.2, 0.3, 0.3), bbox_transform=ax.transAxes, borderpad=0)
        for i, grid in enumerate(all_sim_grids):
            diff = profileZMeanTOPAS - grid[xIndex, yIndex, :]
            ax_inset.plot(zRange, diff, label=f'TOPAS - {sim_labels[i]}', color=colors[i], linestyle='-', linewidth=0.9)
        ax_inset.set_xlabel('Z (mm)', fontsize=12)
        # ax_inset.set_yscale('log')
        ax_inset.set_ylabel('Difference (MeV)', fontsize=12)
        ax_inset.tick_params(labelsize=11)
        ax_inset.grid(True)

        plt.tight_layout()
        plt.savefig(f'{savePath}ProfileMeanEnergy_XIndex{xIndex}_YIndex{yIndex}_{mode}.pdf')
        plt.close()

    # --- Execute Plots ---
    # Load mean energy grid
    meanEnergyGridTOPAS = np.load("../../meanEnergyGridTOPAS.npy")
    
    # 1. Z-profile (Bragg Curve)
    data_topas_z = np.sum(data_topas, axis=(0, 1))
    all_sims_z = [np.sum(data, axis=(0, 1)) for data in all_sim_data]
    plot_z_profile(data_topas_z, all_sims_z, labels)

    # 2. Lateral Profiles at key Z-slices
    plot_lateral_profiles(data_topas, all_sim_data, labels)
    
    # 3. Mean Energy Z-Profile at specific (x,y)
    xIndex, yIndex = 25, 25
    plot_mean_energy_z_profile(meanEnergyGridTOPAS, all_mean_grids, labels, xIndex, yIndex)
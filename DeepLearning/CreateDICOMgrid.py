
import os
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import FuncFormatter

# Matplotlib params
params = {
    'xtick.labelsize': 16,    
    'ytick.labelsize': 16,      
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'legend.fontsize': 12
}
pylab.rcParams.update(params)  # Apply changes

# 1. Read HU values from the interpolation file
def read_hu_values_from_file(file_path="InterpolationData.txt"):
    """
    Reads HU values from the interpolation data file.
    Assumes the first column is the HU value.
    Returns a sorted list of *unique* integer HU values.
    """
    hu_values = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    # First column -> float -> int
                    hu = int(float(line.split()[0]))
                    hu_values.append(hu)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it exists.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    # unique + sorted
    return sorted(list(set(hu_values)))


def pick_n_hus(hu_list, n=50, even_spacing=True):
    """
    Given a sorted list of unique HU values, pick `n` of them.
    If `even_spacing` is True (default), we choose evenly spaced
    indices across the whole list (i.e. uniform sampling).
    Otherwise we simply take the first `n` values.
    """
    if len(hu_list) < n:
        raise ValueError(f"Need at least {n} distinct HU values in InterpolationData.txt")

    if even_spacing:
        indices = np.round(np.linspace(0, len(hu_list) - 1, num=n)).astype(int)
        return [hu_list[i] for i in indices]
    else:
        return hu_list[:n]

# 2. Configuration for the 3-D volume
n_slices = 300       # Z (n_z)
n_rows = 50          # Y
n_cols = 50          # X
pixel_spacing = (4.0, 4.0)  # mm (X,Y)
slice_thickness = 1.0        # mm

# 3. Pick HU values and build slice-to-HU mapping
hu_list = read_hu_values_from_file("InterpolationData.txt")
if not hu_list:
    raise SystemExit("Could not obtain HU values – aborting.")

# Pick 25 evenly spaced HUs for the first half (150 slices)
n = 25
first_half_hus = pick_n_hus(hu_list, n=n, even_spacing=True)
print("Selected HU values for first half:", first_half_hus)

# 3a. Determine slice ranges
slice_groups = []
half_slices = n_slices // 2
group_size = half_slices // n
print("Group size:", group_size)
remainder = half_slices % n

# Determine the group sizes for the first half, adding remainder to the last groups
group_sizes_first_half = [group_size] * n
for i in range(remainder):
    group_sizes_first_half[n - 1 - i] += 1

# The group sizes for the second half are the reverse of the first half
group_sizes_second_half = list(reversed(group_sizes_first_half))
all_group_sizes = group_sizes_first_half + group_sizes_second_half

start = 0
for size in all_group_sizes:
    end = start + size
    slice_groups.append((start, end))
    start = end

# print("Slice ranges (0-based, end-exclusive):", slice_groups)

# 4.  Helper: create a single slice
def make_slice(z_index, hu_value):
    """
    Build a single CT slice with constant HU value.
    """
    ds = Dataset()
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.PatientName = "Test^Patient"
    ds.PatientID   = "123456"

    # Image geometry
    ds.Rows = n_rows
    ds.Columns = n_cols
    ds.PixelSpacing = [str(pixel_spacing[0]), str(pixel_spacing[1])]
    ds.SliceThickness = str(slice_thickness)
    ds.SpatialResolution = "1.0\\1.0"
    ds.SpacingBetweenSlices = str(slice_thickness)
    ds.ImagePositionPatient = [0.0, 0.0, z_index * slice_thickness]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # standard

    # Pixel data – 16‑bit unsigned, all equal to `hu_value`
    pixel_array = np.full((n_rows, n_cols), hu_value, dtype=np.uint16)
    ds.PixelData = pixel_array.tobytes()

    # Required tags
    ds.BitsAllocated = 16
    ds.BitsStored    = 16
    ds.HighBit       = 15
    ds.PixelRepresentation = 0  # unsigned
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"

    # File meta
    ds.file_meta = Dataset()
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.ImplementationClassUID = generate_uid()

    # Time stamps
    dt = datetime.now()
    ds.InstanceCreationDate = dt.strftime("%Y%m%d")
    ds.InstanceCreationTime = dt.strftime("%H%M%S.%f")

    ds.is_little_endian = True
    ds.is_implicit_VR   = False   # ExplicitVRLittleEndian
    return ds

# 5.  Write all slices
out_dir = "dicom_data"
os.makedirs(out_dir, exist_ok=True)

print("[INFO] Writing slices...")
for z in range(n_slices):
    # Find which group this slice belongs to
    group_index = None
    for g_idx, (start, end) in enumerate(slice_groups):
        if start <= z < end:
            group_index = g_idx
            break
    if group_index < n:
        hu_idx = group_index
    else:
        hu_idx = (n - 1) - (group_index - n)

    # hu_idx = 15
    hu_value = first_half_hus[hu_idx]
    # print(f"Slice {z+1:03d} belongs to group {group_index+1}, with HU index {hu_idx} (HU value: {hu_value})")

    ds = make_slice(z, hu_value)
    ds.PixelData = np.full((n_rows, n_cols), hu_value, dtype=np.uint16).tobytes()
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"

    # File name
    filename = os.path.join(out_dir, f"CT_slice_{z+1:03d}.dcm")
    ds.save_as(filename, write_like_original=False)

print(f"[INFO] All {n_slices} slices written to '{out_dir}'.")

# Read the generated DICOM files and visualize the 2D projections
print("\n[INFO] Reading generated DICOM files and creating 2D projections...")
    
# Create an empty 3D NumPy array to hold all pixel data
# The shape is (Z, Y, X)
full_volume = np.zeros((n_slices, n_rows, n_cols), dtype=np.uint16)

# Read each DICOM file and populate the 3D array
for z in range(n_slices):
    filename = os.path.join(out_dir, f"CT_slice_{z+1:03d}.dcm")
    try:
        ds = pydicom.dcmread(filename)
        full_volume[z, :, :] = ds.pixel_array
    except Exception as e:
        print(f"Could not read file {filename}: {e}")
        full_volume = None
        break
    

if full_volume is not None:
    # Get the middle row and column for the projections
    mid_row = n_rows // 2
    mid_col = n_cols // 2
        
    # Extract the Z-Y and Z-X planes from the 3D volume
    zy_plane = full_volume[:, :, mid_col]
    zx_plane = full_volume[:, mid_row, :]
    
    zy_plane_transposed = zy_plane.T
    zx_plane_transposed = zx_plane.T
    
    x_range = [-100., 100.]
    y_range = [-100., 100.]
    z_range = [-150., 150.]
        
    # Define the extents for imshow [left, right, bottom, top]
    extent_zy = [z_range[0], z_range[1], y_range[0], y_range[1]]
    extent_zx = [z_range[0], z_range[1], x_range[0], x_range[1]]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Define the custom formatter for the colorbar
    hu_formatter = FuncFormatter(lambda x, pos: f"{int(x - 1000)}")
        
    # Plot the Z-Y projection
    im1 = ax1.imshow(zy_plane_transposed, cmap='gray', aspect='auto', extent=extent_zy)
    ax1.set_title('Z-Y Plane Projection')
    ax1.set_xlabel('Z (mm)')
    ax1.set_ylabel('Y (mm)')
    # --- Corrected Colorbar Creation ---
    cb1 = fig.colorbar(im1, ax=ax1, label='HU Value')
    cb1.formatter = hu_formatter
    cb1.update_ticks()
        
    # Plot the Z-X projection
    im2 = ax2.imshow(zx_plane_transposed, cmap='gray', aspect='auto', extent=extent_zx)
    ax2.set_title('Z-X Plane Projection')
    ax2.set_xlabel('Z (mm)')
    ax2.set_ylabel('X (mm)')
    # --- Corrected Colorbar Creation ---
    cb2 = fig.colorbar(im2, ax=ax2, label='HU Value')
    cb2.formatter = hu_formatter
    cb2.update_ticks()
        
    plt.tight_layout()
    plt.savefig('PlaneProjectionsDICOM.pdf', dpi=300)
    plt.close()
        
    print("[INFO] Successfully created and saved 'PlaneProjectionsDICOM.pdf'.")
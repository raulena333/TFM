import os
import numpy as np
import pydicom

def load_ct_volume(dicom_dir: str,
                   sort_by: str = 'z',           # 'z' (default) or 'filename'
                   dtype: np.dtype = np.int16):   # dtype for the final array
    """
    Read all DICOM files in *dicom_dir* and return a 3‑D numpy array of HU values.

    Parameters
    ----------
    dicom_dir : str
        Path to the folder that contains the DICOM series.
    sort_by : str, optional
        'z'   – sort by ImagePositionPatient[2] (recommended for CT).
        'filename' – lexicographic order of the file names.
    dtype : np.dtype, optional
        Desired dtype of the returned array (default int16).

    Returns
    -------
    volume : np.ndarray
        3‑D array with shape (Rows, Columns, Slices).  HUs are signed integers.
    """
    # Gather all DICOM file paths
    dicom_paths = [os.path.join(dicom_dir, f)
                   for f in os.listdir(dicom_dir)
                   if f.lower().endswith('.dcm')]

    if not dicom_paths:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

    # Load a few slices to discover image geometry
    sample_ds = pydicom.dcmread(dicom_paths[0])
    rows, cols = int(sample_ds.Rows), int(sample_ds.Columns)

    # Build a list of (path, z‑index) tuples
    slices = []
    for path in dicom_paths:
        ds = pydicom.dcmread(path, stop_before_pixels=True)  # just read meta
        # Use the z‑coordinate if available; otherwise fallback to file name
        if sort_by == 'z':
            # Some series use ImagePositionPatient; others use InstanceNumber.
            z = float(ds.get('ImagePositionPatient', [0, 0, 0])[2])
        else:
            z = float(os.path.basename(path).split('.')[0])  # crude fallback
        slices.append((path, z))

    # Sort slices
    slices.sort(key=lambda x: x[1])

    # Allocate the volume array
    n_slices = len(slices)
    volume = np.empty((rows, cols, n_slices), dtype=dtype)
    
    # Loop over the sorted slices, read pixel data, apply rescaling
    for idx, (path, _) in enumerate(slices):
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)  # keep float until rescale

        # Rescale to HUs if the series includes it
        intercept = float(ds.get('RescaleIntercept', 0.0))
        slope     = float(ds.get('RescaleSlope', 1.0))
        img = img * slope + intercept

        # Clip to signed 16‑bit range (optional)
        img = np.clip(img, -32768, 32767)

        # Store in the volume
        volume[:, :, idx] = img.astype(dtype)

    return volume


if __name__ == '__main__':
    volume = load_ct_volume('dicom_data', sort_by='z')
    print(volume.shape)
    print(volume.dtype)
    print(volume.min(), volume.max())
    print(volume[25, 25, :])
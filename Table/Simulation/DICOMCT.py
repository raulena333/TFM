import numpy as np
import os
import pydicom
from pydicom.dataset import Dataset, FileDataset
import datetime
import time
import SimpleITK as sitk

# Load material grid
materialGrid = np.load("materialGrid.npy")

# Map to HU values
HU_map = {
    0: -700,   # Lung
    1: 0,      # Water
    2: 1200,   # Bone
    3: 100     # Soft tissue
}
HU_grid = np.vectorize(HU_map.get)(materialGrid).astype(np.int16)

# If original shape is (50, 50, 300) => X, Y, Z
# We need to transpose it to (300, 50, 50) => Z, Y, X
HU_grid = np.transpose(HU_grid, (2, 1, 0))  # Now correct for DICOM

# Image parameters
spacing = [4.0, 4.0, 1.0]  # X, Y, Z mm
origin = [0.0, 0.0, 0.0]
pixel_spacing = [str(spacing[1]), str(spacing[0])]  # Row, Column spacing

output_dir = "dicom_ct"
os.makedirs(output_dir, exist_ok=True)

# Time info for DICOM metadata
current_time = time.time()
dt = datetime.datetime.fromtimestamp(current_time)
study_date = dt.strftime('%Y%m%d')
study_time = dt.strftime('%H%M%S')

# Create slices
num_slices = HU_grid.shape[0]
rows, cols = HU_grid.shape[1], HU_grid.shape[2]

for z in range(num_slices):
    filename = os.path.join(output_dir, f"slice_{z:04d}.dcm")
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "Test^Phantom"
    ds.PatientID = "123456"
    ds.Modality = "CT"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

    ds.ImagePositionPatient = [str(origin[0]), str(origin[1]), str(origin[2] + z * spacing[2])]
    ds.ImageOrientationPatient = ['1', '0', '0', '0', '1', '0']
    ds.PixelSpacing = pixel_spacing
    ds.SliceThickness = str(spacing[2])
    ds.Rows = rows
    ds.Columns = cols
    ds.InstanceNumber = z + 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed int
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.StudyDate = study_date
    ds.StudyTime = study_time

    slice_data = HU_grid[z, :, :]
    ds.PixelData = slice_data.tobytes()

    ds.save_as(filename)

print(f"DICOM series written to: {output_dir}")


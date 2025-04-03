import h5py
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

""" CUDA Kernel
1. **Define the Kernel (`cuda_kernel`)**:  
   - The CUDA function `square()` is written as a string in C-style.
   - `__global__` makes it a GPU function.
   - Each thread computes `d_out[idx] = d_in[idx] * d_in[idx]`.

2. **Compile with `SourceModule`**:  
   - `mod = SourceModule(cuda_kernel)` compiles the kernel.
   - `mod.get_function("square")` retrieves the compiled function.

3. **Allocate and Transfer Memory**:  
   - `cuda.mem_alloc()` allocates GPU memory.
   - `cuda.memcpy_htod()` copies CPU data to GPU.

4. **Launch the Kernel**:  
   - `square_kernel(d_output, d_input, block=(block_size, 1, 1), grid=(grid_size, 1, 1))` runs the function on the GPU.

5. **Retrieve Results**:  
   - `cuda.memcpy_dtoh()` copies data back to CPU.
   
"""

# "C:\Program Files\Microsoft Visual Studio\2022\Preview\VC\Auxiliary\Build\vcvars64.bat"
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

# CUDA Kernel (simple example: square each element)
cuda_kernel = """
__global__ void square_array(float *d_data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_data[idx] *= d_data[idx]; // Square the element
    }
}
"""

def process_on_cuda(data):
    data = data.astype(np.float32)  # Ensure it's float32 for CUDA
    size = data.nbytes  # Get size in bytes

    # Allocate GPU memory
    d_data = cuda.mem_alloc(size)

    # Copy data from CPU to GPU
    cuda.memcpy_htod(d_data, data)

    # Compile and get kernel function
    mod = SourceModule(cuda_kernel)
    square_array = mod.get_function("square_array")

    # Launch kernel (assume 256 threads per block)
    block_size = 256
    grid_size = (len(data) + block_size - 1) // block_size  # Ensure all elements are covered
    square_array(d_data, np.int32(len(data)), block=(block_size, 1, 1), grid=(grid_size, 1))

    # Copy result back to CPU
    result = np.empty_like(data)
    cuda.memcpy_dtoh(result, d_data)

    return result

# Example Usage
hdf5_path = "4DTableEnergy.h5"
npz_path = "4DTable.npz"
material = "G4_WATER" 
energy = 200.0 

energyVec_hdf5, angleVec_hdf5, energyVec_npz, angleVec_npz = load_hdf5_npz(hdf5_path, npz_path, material, energy)

# Process HDF5 energy vector on CUDA
processed_energy_hdf5 = process_on_cuda(energyVec_hdf5)

print("Original HDF5 Energy Vector:", energyVec_hdf5)
print("Processed HDF5 Energy Vector (Squared):", processed_energy_hdf5)

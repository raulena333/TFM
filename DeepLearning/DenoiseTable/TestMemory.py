#!/usr/bin/env python3
"""
load_and_report_npz.py

Loads an entire .npz archive into RAM and prints a detailed memory‑usage report.
"""

import os
import sys
import numpy as np
import psutil

# ------------------------------------------------------------------
# Helper: pretty‑print a byte value
# ------------------------------------------------------------------
def pretty_bytes(b: int) -> str:
    """Return a human‑readable string for a number of bytes."""
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB']
    for unit in units:
        if b < 1024:
            return f"{b:,.2f} {unit}"
        b /= 1024.0
    return f"{b:,.2f} EB"

# ------------------------------------------------------------------
# Main routine
# ------------------------------------------------------------------
def main(npz_path: str) -> None:
    if not os.path.isfile(npz_path):
        print(f"File not found: {npz_path}")
        sys.exit(1)

    print(f"\nLoading '{npz_path}' into RAM …")
    npz = np.load(npz_path, allow_pickle=False)          # full load
    
    print(f"\n{len(npz.files)} array(s) in the archive:\n")
    print(f"{'Key':>15} | {'Shape':>20} | {'Dtype':>8} | {'Size'}")
    print("-" * 70)

    total_bytes = 0
    for key in npz.files:
        arr = npz[key]
        bytes_ = arr.nbytes
        total_bytes += bytes_
        print(f"{key:>15} | {str(arr.shape):>20} | {arr.dtype} | {pretty_bytes(bytes_)}")

    print("\n" + "=" * 70)
    print(f"Total memory needed for all arrays (if fully loaded): {pretty_bytes(total_bytes)}")

    # Optional: report actual process memory
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        print("\n Process memory after load:")
        print(f" RSS (resident): {pretty_bytes(mem_info.rss)}")
        print(f" VMS (virtual):  {pretty_bytes(mem_info.vms)}")
    except ImportError:
        # psutil not installed – silently skip
        pass

# ------------------------------------------------------------------
# CLI support
# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python3 TestMemory.py <path/to/archive.npz>")
        sys.exit(1)

    main(sys.argv[1])
    
    mem = psutil.virtual_memory()
    print(f"Total physical RAM: {mem.total / 1024 ** 3:,.2f} GB")
    print(f"Available (unused) RAM: {mem.available / 1024 ** 3:,.2f} GB")
    
    



"""
SW-ADP Setup Verification Script
Confirms CUDA availability, PyTorch GPU access, and all required imports.

Usage:
    conda activate sw-adp
    python scripts/verify_setup.py
"""

import sys
import importlib

REQUIRED_PACKAGES = [
    ("torch", "PyTorch"),
    ("sklearn", "scikit-learn"),
    ("stumpy", "STUMPY"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("shap", "SHAP"),
    ("tqdm", "tqdm"),
    ("scipy", "SciPy"),
    ("numba", "Numba"),
]

def check_imports():
    """Check all required packages are importable."""
    print("=" * 60)
    print("SW-ADP ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print(f"\nPython: {sys.version}")
    print(f"Executable: {sys.executable}\n")

    all_ok = True
    for module_name, display_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK]   {display_name:20s} {version}")
        except ImportError as e:
            print(f"  [FAIL] {display_name:20s} -- {e}")
            all_ok = False

    return all_ok


def check_cuda():
    """Check CUDA availability and GPU details."""
    print("\n" + "-" * 60)
    print("CUDA / GPU CHECK")
    print("-" * 60)

    try:
        import torch
    except ImportError:
        print("  [FAIL] PyTorch not installed, cannot check CUDA.")
        return False

    cuda_available = torch.cuda.is_available()
    print(f"\n  torch.cuda.is_available(): {cuda_available}")
    print(f"  PyTorch CUDA version:      {torch.version.cuda or 'N/A'}")
    print(f"  cuDNN version:             {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"  GPU count:                 {device_count}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            mem_total = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
            print(f"  GPU {i}: {name} ({mem_total:.1f} GB)")

        # Quick tensor test on GPU
        try:
            x = torch.randn(256, 256, device="cuda")
            y = x @ x.T
            del x, y
            torch.cuda.empty_cache()
            print("\n  [OK]   GPU tensor operation successful")
        except Exception as e:
            print(f"\n  [FAIL] GPU tensor operation failed: {e}")
            return False
    else:
        print("\n  [WARN] CUDA not available. Training will be CPU-only (very slow).")
        print("         Verify your NVIDIA driver and CUDA toolkit installation.")
        print("         Expected GPU: RTX 4060Ti 8GB")
        return False

    return True


def check_stumpy():
    """Quick STUMPY sanity check (depends on numba JIT)."""
    print("\n" + "-" * 60)
    print("STUMPY SANITY CHECK")
    print("-" * 60)

    try:
        import stumpy
        import numpy as np
        ts = np.random.rand(200)
        mp = stumpy.stump(ts, m=20)
        print(f"  [OK]   STUMPY matrix profile computed (shape: {mp.shape})")
        return True
    except Exception as e:
        print(f"  [FAIL] STUMPY test failed: {e}")
        return False


def main():
    imports_ok = check_imports()
    cuda_ok = check_cuda()
    stumpy_ok = check_stumpy()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Imports:  {'PASS' if imports_ok else 'FAIL'}")
    print(f"  CUDA/GPU: {'PASS' if cuda_ok else 'FAIL'}")
    print(f"  STUMPY:   {'PASS' if stumpy_ok else 'FAIL'}")

    if imports_ok and cuda_ok and stumpy_ok:
        print("\n  All checks passed. Environment is ready.")
        print("  Next step: python scripts/download_opssat.py")
        return 0
    else:
        print("\n  Some checks failed. Fix issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

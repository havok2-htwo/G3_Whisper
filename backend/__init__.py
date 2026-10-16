"""Backend package for the GENESIS Whisper server."""

import os

# Number CUDA devices like nvidia-smi / NVML (PCI bus order), so local_gpu_device "cuda:1" is the card nvidia-smi
# calls GPU 1. CUDA's default order is "fastest first", which differs on mixed boxes (e.g. 3090 + 2070S). Set here,
# in the package init, because it must happen before torch initialises CUDA and several backend modules import torch
# at the top; an explicit environment value wins.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

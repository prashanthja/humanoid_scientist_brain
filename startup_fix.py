# Add this at the TOP of learning_module/trainer_online.py
# Replace the existing device detection with this:

import torch

def get_device():
    """Get best available device — MPS on Mac, CUDA on cloud GPU, CPU otherwise."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS only on Apple Silicon — not available on cloud servers
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Test MPS actually works
            test = torch.zeros(1).to("mps")
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")

DEVICE = get_device()

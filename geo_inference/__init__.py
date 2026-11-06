import os

# Let PyTorch run any operator Apple's GPU backend (MPS) lacks on the CPU
# instead of failing. Must be set before torch is imported; no effect elsewhere.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

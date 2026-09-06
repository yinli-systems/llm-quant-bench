#!/usr/bin/env python3
"""Exercise the exact FlashInfer JIT sampling path before expensive model startup."""
import json
import os
import subprocess

import torch
import flashinfer
from flashinfer.jit.cpp_ext import get_cuda_path


def main():
    assert torch.cuda.is_available(), "CUDA unavailable"
    assert torch.cuda.get_device_capability() == (8, 9), "Expected RTX 4090 / SM89"
    assert get_cuda_path() == os.environ["CUDA_HOME"]
    logits = torch.zeros((2, 256), device="cuda", dtype=torch.float32)
    logits[0, 7] = 100
    logits[1, 42] = 100
    sampled = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, 1, 0.99, deterministic=True)
    torch.cuda.synchronize()
    assert sampled.tolist() == [7, 42], sampled.tolist()
    print(json.dumps({
        "status": "passed", "test": "flashinfer_jit_sampling",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "flashinfer": flashinfer.__version__,
        "cuda_home": get_cuda_path(), "sampled": sampled.tolist(),
        "nvcc": subprocess.check_output([get_cuda_path() + "/bin/nvcc", "--version"], text=True),
    }, indent=2))


if __name__ == "__main__":
    main()

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # has to be set before importing torch

import sys
import time

import coremltools as ct
import timm
import torch


def main():
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("timm:", timm.__version__)
    print("coremltools:", ct.__version__)
    print("mps available:", torch.backends.mps.is_available())

    torch.manual_seed(0)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=200).eval()
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        ref = model(x)  # cpu reference
        model.to("mps")
        t0 = time.perf_counter()
        out = model(x.to("mps"))
        torch.mps.synchronize()  # mps is async, wait before reading the clock
        dt = time.perf_counter() - t0

    print("output:", tuple(out.shape))
    print(f"mps forward: {dt * 1000:.0f} ms (first call, not a benchmark)")
    print(f"max diff vs cpu: {(out.cpu() - ref).abs().max().item():.2e}")


if __name__ == "__main__":
    main()

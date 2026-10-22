"""GPU benchmark helpers."""
import time
import numpy as np
import torch

def timed(fn, dev, reps=3):
    fn()
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(reps):
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(dev)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), torch.cuda.max_memory_allocated(dev) / 2**30

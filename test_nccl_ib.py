"""Quick test: verify NCCL uses InfiniBand across 2 nodes."""
import os
import time
import torch
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"NCCL version: {torch.cuda.nccl.version()}")
        for k in sorted(os.environ):
            if "NCCL" in k or "IB" in k:
                print(f"  {k}={os.environ[k]}")

    # Warmup
    t = torch.ones(1, device="cuda") * rank
    dist.all_reduce(t)
    torch.cuda.synchronize()

    # Bandwidth test: 256MB all-reduce
    size = 256 * 1024 * 1024 // 4  # 256MB of float32
    data = torch.randn(size, device="cuda")
    torch.cuda.synchronize()
    dist.barrier()

    start = time.time()
    for _ in range(10):
        dist.all_reduce(data)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    if rank == 0:
        bw = (size * 4 * 2 * 10) / elapsed / 1e9  # algobw in GB/s
        print(f"\nAll-reduce 256MB x 10 iters: {elapsed:.2f}s")
        print(f"Algo bandwidth: {bw:.1f} GB/s")
        if bw > 10:
            print("✓ Likely using InfiniBand (high bandwidth)")
        elif bw > 1:
            print("? Moderate bandwidth - might be Ethernet or slow IB")
        else:
            print("✗ Very low bandwidth - likely TCP/Ethernet fallback")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

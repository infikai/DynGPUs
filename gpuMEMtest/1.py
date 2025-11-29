import torch

# 1. Initialize CUDA (creates the context)
x = torch.tensor([1.0]).cuda()

# 2. Get raw data
total_memory = torch.cuda.get_device_properties(0).total_memory
reserved = torch.cuda.memory_reserved(0)
allocated = torch.cuda.memory_allocated(0)

# 3. Get total used from driver (requires pynvml or crude calculation)
# Note: torch.cuda.mem_get_info() returns (free, total)
free_mem, total_mem = torch.cuda.mem_get_info(0)
used_mem_driver = total_mem - free_mem

# 4. Calculate Breakdown
context_overhead = used_mem_driver - reserved
fragmentation = reserved - allocated

print(f"Total GPU Memory: {total_mem / 1024**2:.2f} MB")
print(f"Driver/Context Overhead: {context_overhead / 1024**2:.2f} MB")
print(f"PyTorch Reserved (Cached): {reserved / 1024**2:.2f} MB")
print(f" - Actual Tensors: {allocated / 1024**2:.2f} MB")
print(f" - Fragmentation/Cache: {fragmentation / 1024**2:.2f} MB")
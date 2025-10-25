import os, time, argparse, random

def write_read_fixed_total(path, block_kb=64, total_mb=1):
    """Test I/O performance with fixed total data size but varying block sizes.
    
    This properly tests the effect of block size on I/O efficiency by:
    - Keeping total data size constant
    - Varying block size (and thus number of operations)
    - Measuring the impact of system call overhead vs data transfer efficiency
    """
    total_bytes = total_mb * 1024 * 1024
    block_bytes = block_kb * 1024
    num_blocks = total_bytes // block_bytes
    
    # Generate random data for the block
    block_data = os.urandom(block_bytes)
    
    t0 = time.perf_counter()
    
    # Write phase: write num_blocks files, each of block_kb size
    file_paths = []
    for i in range(num_blocks):
        file_path = f"{path}/block_{i}.bin"
        file_paths.append(file_path)
        with open(file_path, "wb") as f:
            f.write(block_data)
    
    # Read phase: read all files back
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            _ = f.read()
    
    # Cleanup phase: remove all files
    for file_path in file_paths:
        os.remove(file_path)
    
    t1 = time.perf_counter()
    return (t1-t0)*1000.0

def write_read_fixed_operations(path, block_kb=64, operations=20):
    """Test I/O performance with fixed number of operations but varying block sizes.
    
    This tests the raw efficiency of different block sizes by:
    - Keeping number of I/O operations constant
    - Varying data size per operation
    - Measuring pure I/O throughput
    """
    block_bytes = block_kb * 1024
    block_data = os.urandom(block_bytes)
    
    t0 = time.perf_counter()
    
    # Write phase
    file_paths = []
    for i in range(operations):
        file_path = f"{path}/op_{i}.bin"
        file_paths.append(file_path)
        with open(file_path, "wb") as f:
            f.write(block_data)
    
    # Read phase
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            _ = f.read()
    
    # Cleanup phase
    for file_path in file_paths:
        os.remove(file_path)
    
    t1 = time.perf_counter()
    return (t1-t0)*1000.0

def write_read_legacy(path, size_kb=64, count=20):
    """Legacy version - kept for backward compatibility."""
    data = os.urandom(size_kb*1024)
    t0 = time.perf_counter()
    for i in range(count):
        p = f"{path}/blk_{i}.bin"
        with open(p, "wb") as f:
            f.write(data)
        with open(p, "rb") as f:
            _ = f.read()
        os.remove(p)
    t1 = time.perf_counter()
    return (t1-t0)*1000.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="I/O Performance Benchmark Test")
    ap.add_argument("--dir", default="./tmpio", help="Temporary directory for test files")
    ap.add_argument("--block_kb", type=int, default=64, help="Block size in KB")
    ap.add_argument("--total_mb", type=int, default=1, help="Total data size in MB (for fixed_total mode)")
    ap.add_argument("--operations", type=int, default=20, help="Number of operations (for fixed_operations mode)")
    ap.add_argument("--mode", choices=["fixed_total", "fixed_operations", "legacy"], 
                   default="fixed_total", help="Test mode")
    args = ap.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    
    # Choose test mode
    if args.mode == "fixed_total":
        ms = write_read_fixed_total(args.dir, args.block_kb, args.total_mb)
        num_blocks = (args.total_mb * 1024 * 1024) // (args.block_kb * 1024)
        print(f"wall_ms={ms:.3f},block_kb={args.block_kb},total_mb={args.total_mb},blocks={num_blocks}")
    elif args.mode == "fixed_operations":
        ms = write_read_fixed_operations(args.dir, args.block_kb, args.operations)
        total_mb = (args.block_kb * args.operations) / 1024
        print(f"wall_ms={ms:.3f},block_kb={args.block_kb},operations={args.operations},total_mb={total_mb:.2f}")
    else:  # legacy mode
        ms = write_read_legacy(args.dir, args.block_kb, args.operations)
        print(f"wall_ms={ms:.3f},block_kb={args.block_kb},count={args.operations}")

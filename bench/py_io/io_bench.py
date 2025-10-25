import os, time, argparse, random
def write_read(path, size_kb=64, count=20):  # Reduced for home computer compatibility
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

def write_read_optimized(path, size_kb=64, count=20):
    """Optimized version that reduces file system overhead."""
    data = os.urandom(size_kb*1024)
    t0 = time.perf_counter()
    
    # Pre-create all files to reduce overhead
    file_paths = [f"{path}/blk_{i}.bin" for i in range(count)]
    
    # Write phase
    for p in file_paths:
        with open(p, "wb") as f:
            f.write(data)
    
    # Read phase  
    for p in file_paths:
        with open(p, "rb") as f:
            _ = f.read()
    
    # Cleanup phase
    for p in file_paths:
        os.remove(p)
    
    t1 = time.perf_counter()
    return (t1-t0)*1000.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./tmpio")
    ap.add_argument("--block_kb", type=int, default=64)
    ap.add_argument("--count", type=int, default=20)  # Reduced for home computer compatibility
    ap.add_argument("--optimized", action="store_true", help="Use optimized I/O test")
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    
    # Use optimized version if requested
    if args.optimized:
        ms = write_read_optimized(args.dir, args.block_kb, args.count)
    else:
        ms = write_read(args.dir, args.block_kb, args.count)
    
    print(f"wall_ms={ms:.3f},block_kb={args.block_kb},count={args.count}")

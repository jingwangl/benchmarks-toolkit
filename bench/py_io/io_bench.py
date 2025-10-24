import os, time, argparse, random
def write_read(path, size_kb=64, count=200):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="./tmpio")
    ap.add_argument("--block_kb", type=int, default=64)
    ap.add_argument("--count", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    ms = write_read(args.dir, args.block_kb, args.count)
    print(f"wall_ms={ms:.3f},block_kb={args.block_kb},count={args.count}")
